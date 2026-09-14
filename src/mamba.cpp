// src/mamba.cpp
#include "llm/mamba.h"
#include "llm/mamba_scan.h"
#include <cmath>
#include <cassert>
#include <algorithm>

namespace llm {

MambaBlock::MambaBlock(size_t n_embd, size_t d_inner, size_t d_state,
                       size_t dt_rank, size_t conv_kernel, bool use_rmsnorm)
    : n_embd_(n_embd),
      d_inner_(d_inner),
      d_state_(d_state),
      dt_rank_(dt_rank),
      conv_kernel_(conv_kernel),
      use_rmsnorm_(use_rmsnorm),
      norm_weight_({n_embd}, 1.0f),
      in_proj_weight_({2 * d_inner, n_embd}),
      conv1d_weight_({d_inner, 1, conv_kernel}),
      conv1d_bias_({d_inner}),
      x_proj_weight_({dt_rank + 2 * d_state, d_inner}),
      dt_proj_weight_({d_inner, dt_rank}),
      a_log_({d_inner, d_state}),
      d_weight_({d_inner}, 1.0f),
      out_proj_weight_({n_embd, d_inner}) {
    // Initialize parameters
    in_proj_weight_.randn(0, 1.0f / std::sqrt((float)n_embd));
    conv1d_weight_.randn(0, 1.0f / std::sqrt((float)(d_inner * conv_kernel)));
    conv1d_bias_.fill(0.0f);
    x_proj_weight_.randn(0, 1.0f / std::sqrt((float)d_inner));
    dt_proj_weight_.randn(0, 0.001f);
    a_log_.fill(-1.0f);
    conv_state_.assign(d_inner * (conv_kernel - 1), 0.0f);
}

Tensor MambaBlock::conv1d_forward(const Tensor& x) const {
    // x: [T, d_inner], conv1d_weight: [d_inner, 1, conv_kernel]
    // Depth-wise: each channel has its own filter
    size_t T = x.shape[0];
    size_t D = d_inner_;
    size_t k = conv_kernel_;
    Tensor out({T, D}, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < D; ++d) {
            float acc = conv1d_bias_.data[d];
            for (size_t j = 0; j < k; ++j) {
                ssize_t t_in = (ssize_t)t - (ssize_t)(k - 1) + (ssize_t)j;
                if (t_in >= 0) {
                    // conv1d_weight_ is [d_inner, 1, conv_kernel] stored as flat
                    size_t w_idx = d * k + j;
                    acc += x(t_in, d) * conv1d_weight_.data[w_idx];
                }
            }
            out(t, d) = acc;
        }
    }
    return out;
}

Tensor MambaBlock::forward(const Tensor& x) const {
    assert(x.shape[1] == n_embd_);
    size_t T = x.shape[0];

    // 1. RMSNorm
    Tensor normed = use_rmsnorm_ ? x.rmsnorm(&norm_weight_) : x.layernorm(&norm_weight_);

    // 2. Input projection: [T, 2*d_inner]
    Tensor proj = normed.matmul(in_proj_weight_.transpose());
    assert(proj.shape[0] == T && proj.shape[1] == 2 * d_inner_);

    // Split into gate (x_z) and SSM branch (x_ssm)
    Tensor x_z({T, d_inner_}, 0.0f);
    Tensor x_ssm({T, d_inner_}, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_inner_; ++d) {
            x_z(t, d) = proj(t, d);
            x_ssm(t, d) = proj(t, d + d_inner_);
        }
    }

    // 3. Conv1D + SiLU on x_ssm
    Tensor conv_out = conv1d_forward(x_ssm);
    for (auto& v : conv_out.data) v = v / (1.0f + std::exp(-v));

    // 4. SSM parameters: dt, B, C
    Tensor x_proj_out = conv_out.matmul(x_proj_weight_.transpose());
    // x_proj_out: [T, dt_rank + 2*d_state]
    Tensor dt({T, d_inner_}, 0.0f);
    Tensor B({T, d_state_}, 0.0f);
    Tensor C({T, d_state_}, 0.0f);

    for (size_t t = 0; t < T; ++t) {
        // dt: dt_proj then softplus
        for (size_t d = 0; d < d_inner_; ++d) {
            float dt_raw = 0.0f;
            for (size_t r = 0; r < dt_rank_; ++r) {
                dt_raw += x_proj_out(t, r) * dt_proj_weight_(d, r);
            }
            dt(t, d) = std::log(1.0f + std::exp(dt_raw));
            dt(t, d) = std::max(0.001f, std::min(dt(t, d), 0.1f));
        }
        // B: [d_state]
        for (size_t n = 0; n < d_state_; ++n) {
            B(t, n) = x_proj_out(t, dt_rank_ + n);
        }
        // C: [d_state]
        for (size_t n = 0; n < d_state_; ++n) {
            C(t, n) = x_proj_out(t, dt_rank_ + d_state_ + n);
        }
    }

    // 5. Selective scan
    std::vector<float> x_flat(T * d_inner_);
    std::vector<float> A_flat(d_inner_ * d_state_);
    std::vector<float> B_flat(T * d_state_);
    std::vector<float> C_flat(T * d_state_);
    std::vector<float> dt_flat(T * d_inner_);
    std::vector<float> D_flat(d_inner_);
    std::vector<float> h0(d_inner_ * d_state_, 0.0f);

    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < d_inner_; ++d)
            x_flat[t * d_inner_ + d] = conv_out(t, d);
    for (size_t d = 0; d < d_inner_; ++d)
        for (size_t n = 0; n < d_state_; ++n)
            A_flat[d * d_state_ + n] = a_log_(d, n);
    for (size_t t = 0; t < T; ++t)
        for (size_t n = 0; n < d_state_; ++n) {
            B_flat[t * d_state_ + n] = B(t, n);
            C_flat[t * d_state_ + n] = C(t, n);
        }
    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < d_inner_; ++d)
            dt_flat[t * d_inner_ + d] = dt(t, d);
    for (size_t d = 0; d < d_inner_; ++d) D_flat[d] = d_weight_.data[d];

    std::vector<float> y_flat(T * d_inner_);
    selective_scan_sequential(x_flat, A_flat, B_flat, C_flat, dt_flat, D_flat, h0, y_flat, T, d_inner_, d_state_);

    Tensor ssm_out({T, d_inner_}, 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < d_inner_; ++d)
            ssm_out(t, d) = y_flat[t * d_inner_ + d];

    // 6. Gate: SiLU(x_z) * ssm_out
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_inner_; ++d) {
            float gate = x_z(t, d) / (1.0f + std::exp(-x_z(t, d)));
            ssm_out(t, d) = gate * ssm_out(t, d);
        }
    }

    // 7. Output projection
    Tensor out = ssm_out.matmul(out_proj_weight_.transpose());

    // 8. Residual
    for (size_t i = 0; i < x.data.size(); ++i) {
        out.data[i] = x.data[i] + out.data[i];
    }

    return out;
}

Tensor MambaBlock::backward(const Tensor& x, const Tensor& grad_out) const {
    return Tensor(x.shape, 0.0f);
}

std::vector<Tensor*> MambaBlock::parameters() {
    return {
        &norm_weight_, &in_proj_weight_, &conv1d_weight_, &conv1d_bias_,
        &x_proj_weight_, &dt_proj_weight_, &a_log_, &d_weight_, &out_proj_weight_
    };
}

std::vector<const Tensor*> MambaBlock::parameters() const {
    return {
        &norm_weight_, &in_proj_weight_, &conv1d_weight_, &conv1d_bias_,
        &x_proj_weight_, &dt_proj_weight_, &a_log_, &d_weight_, &out_proj_weight_
    };
}

}  // namespace llm
