// src/mamba.cpp
#include "llm/mamba.h"
#include "llm/mamba_scan.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

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
    if (x.shape[1] != n_embd_) {
        throw std::runtime_error("MambaBlock::forward: input feature dim mismatch");
    }
    size_t T = x.shape[0];

    // 1. RMSNorm
    Tensor normed = use_rmsnorm_ ? x.rmsnorm(&norm_weight_) : x.layernorm(&norm_weight_);

    // 2. Input projection: [T, 2*d_inner]
    Tensor proj = normed.matmul(in_proj_weight_.transpose());
    if (proj.shape[0] != T || proj.shape[1] != 2 * d_inner_) {
        throw std::runtime_error("MambaBlock::forward: projection shape mismatch");
    }

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

static inline float silu_fn(float v) {
    return v / (1.0f + std::exp(-v));
}
static inline float silu_grad(float v) {
    float s = 1.0f / (1.0f + std::exp(-v));
    return s * (1.0f + v * (1.0f - s));
}
static inline float sigmoid_fn(float v) {
    return 1.0f / (1.0f + std::exp(-v));
}

Tensor MambaBlock::backward(const Tensor& x, const Tensor& grad_out) const {
    // Full BPTT through forward(): recompute intermediates, then chain rule in
    // reverse. Accumulates param grads via add_grad, returns dL/dx [T, n_embd].
    size_t T = x.shape[0];
    size_t E = n_embd_, D = d_inner_, N = d_state_, R = dt_rank_, K = conv_kernel_;
    size_t RP = R + 2 * N;

    // ---- forward recompute (mirrors forward()) ----
    Tensor normed = use_rmsnorm_ ? x.rmsnorm(&norm_weight_) : x.layernorm(&norm_weight_);
    Tensor proj = normed.matmul(in_proj_weight_.transpose());  // [T, 2D]
    Tensor x_z({T, D}, 0.0f), x_ssm({T, D}, 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d) {
            x_z(t, d) = proj(t, d);
            x_ssm(t, d) = proj(t, d + D);
        }
    Tensor conv_pre = conv1d_forward(x_ssm);  // pre-SiLU
    Tensor conv_out({T, D}, 0.0f);
    for (size_t i = 0; i < conv_pre.data.size(); ++i) conv_out.data[i] = silu_fn(conv_pre.data[i]);
    Tensor xp_out = conv_out.matmul(x_proj_weight_.transpose());  // [T, RP]
    Tensor dt({T, D}, 0.0f), Bm({T, N}, 0.0f), Cm({T, N}, 0.0f);
    Tensor dt_raw({T, D}, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < D; ++d) {
            float r = 0.0f;
            for (size_t rr = 0; rr < R; ++rr) r += xp_out(t, rr) * dt_proj_weight_(d, rr);
            dt_raw(t, d) = r;
            float sp = std::log(1.0f + std::exp(r));
            dt(t, d) = std::max(0.001f, std::min(sp, 0.1f));
        }
        for (size_t n = 0; n < N; ++n) {
            Bm(t, n) = xp_out(t, R + n);
            Cm(t, n) = xp_out(t, R + N + n);
        }
    }
    // Scan forward with stored trajectory h_all[t][d][n] for BPTT.
    std::vector<float> h_all(T * D * N, 0.0f);
    {
        std::vector<float> h(D * N, 0.0f);
        for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < D; ++d) {
                float xv = conv_out(t, d);
                float dtv = dt(t, d);
                for (size_t n = 0; n < N; ++n) {
                    float a = std::exp(a_log_(d, n) * dtv);
                    h[d * N + n] = a * h[d * N + n] + Bm(t, n) * xv * dtv;
                    h_all[(t * D + d) * N + n] = h[d * N + n];
                }
            }
        }
    }
    Tensor y_ssm({T, D}, 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (size_t n = 0; n < N; ++n) acc += Cm(t, n) * h_all[(t * D + d) * N + n];
            y_ssm(t, d) = acc + d_weight_.data[d] * conv_out(t, d);
        }
    Tensor gate({T, D}, 0.0f);
    for (size_t i = 0; i < gate.data.size(); ++i) gate.data[i] = silu_fn(x_z.data[i]);
    Tensor gated({T, D}, 0.0f);
    for (size_t i = 0; i < gated.data.size(); ++i) gated.data[i] = gate.data[i] * y_ssm.data[i];

    // ---- backward ----
    // out = x + gated @ out_proj^T  => dGated = grad_out @ out_proj
    Tensor dGated({T, D}, 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (size_t e = 0; e < E; ++e) acc += grad_out(t, e) * out_proj_weight_(e, d);
            dGated(t, d) = acc;
        }
    // dOut_proj[e,d] = sum_t grad_out[t,e]*gated[t,d]
    {
        Tensor dW({E, D}, 0.0f);
        for (size_t e = 0; e < E; ++e)
            for (size_t d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (size_t t = 0; t < T; ++t) acc += grad_out(t, e) * gated(t, d);
                dW(e, d) = acc;
            }
        const_cast<Tensor&>(out_proj_weight_).add_grad(dW);
    }
    // gate * y_ssm split
    Tensor dGate({T, D}, 0.0f), dY({T, D}, 0.0f);
    for (size_t i = 0; i < dGated.data.size(); ++i) {
        dGate.data[i] = dGated.data[i] * y_ssm.data[i];
        dY.data[i] = dGated.data[i] * gate.data[i];
    }
    // dX_z = dGate * silu'(x_z)
    Tensor dXz({T, D}, 0.0f);
    for (size_t i = 0; i < dXz.data.size(); ++i) dXz.data[i] = dGate.data[i] * silu_grad(x_z.data[i]);

    // Scan BPTT: reverse over t, per (d,n).
    Tensor dConv_scan({T, D}, 0.0f);  // dx part (+ Dskip direct)
    Tensor dBscan({T, N}, 0.0f), dCscan({T, N}, 0.0f), dDt({T, D}, 0.0f);
    Tensor dA({D, N}, 0.0f);
    Tensor dDskip({D}, 0.0f);
    {
        std::vector<float> dh_next(D * N, 0.0f);
        for (size_t ti = T; ti-- > 0;) {
            size_t t = ti;
            for (size_t d = 0; d < D; ++d) {
                float dy = dY(t, d);
                float xv = conv_out(t, d);
                float dtv = dt(t, d);
                dDskip.data[d] += dy * xv;
                dConv_scan(t, d) += dy * d_weight_.data[d];
                for (size_t n = 0; n < N; ++n) {
                    float h_t = h_all[(t * D + d) * N + n];
                    float h_prev = (t == 0) ? 0.0f : h_all[((t - 1) * D + d) * N + n];
                    float Ctv = Cm(t, n);
                    float Btv = Bm(t, n);
                    float Adn = a_log_(d, n);
                    float a = std::exp(Adn * dtv);
                    float dh = dh_next[d * N + n] + dy * Ctv;
                    dCscan(t, n) += dy * h_t;
                    dBscan(t, n) += dh * xv * dtv;
                    dConv_scan(t, d) += dh * Btv * dtv;
                    // dA via a*h_prev path: dh*h_prev*a*dt
                    dA(d, n) += dh * h_prev * a * dtv;
                    // ddt via h_prev*a*A and B*x paths
                    dDt(t, d) += dh * h_prev * a * Adn + dh * Btv * xv;
                    dh_next[d * N + n] = dh * a;
                }
            }
        }
        const_cast<Tensor&>(a_log_).add_grad(dA);
        const_cast<Tensor&>(d_weight_).add_grad(dDskip);
    }

    // dt clamp mask + softplus backward -> dDt_raw; accumulate into xp path.
    Tensor dXp({T, RP}, 0.0f);
    // B/C columns of xp get scan grads directly.
    for (size_t t = 0; t < T; ++t)
        for (size_t n = 0; n < N; ++n) {
            dXp(t, R + n) += dBscan(t, n);
            dXp(t, R + N + n) += dCscan(t, n);
        }
    Tensor dDtProj({D, R}, 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d) {
            float sp = std::log(1.0f + std::exp(dt_raw(t, d)));
            float in_range = (sp >= 0.001f && sp <= 0.1f) ? 1.0f : 0.0f;
            float d_raw = dDt(t, d) * in_range * sigmoid_fn(dt_raw(t, d));
            for (size_t r = 0; r < R; ++r) {
                dXp(t, r) += d_raw * dt_proj_weight_(d, r);
                dDtProj(d, r) += d_raw * xp_out(t, r);
            }
        }
    const_cast<Tensor&>(dt_proj_weight_).add_grad(dDtProj);
    // dConv from xp path: dConv_xp = dXp @ x_proj
    Tensor dConv_xp({T, D}, 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (size_t j = 0; j < RP; ++j) acc += dXp(t, j) * x_proj_weight_(j, d);
            dConv_xp(t, d) = acc;
        }
    {
        Tensor dW({RP, D}, 0.0f);
        for (size_t j = 0; j < RP; ++j)
            for (size_t d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (size_t t = 0; t < T; ++t) acc += dXp(t, j) * conv_out(t, d);
                dW(j, d) = acc;
            }
        const_cast<Tensor&>(x_proj_weight_).add_grad(dW);
    }
    Tensor dConv({T, D}, 0.0f);
    for (size_t i = 0; i < dConv.data.size(); ++i) dConv.data[i] = dConv_scan.data[i] + dConv_xp.data[i];
    // SiLU backward through conv activation
    Tensor dConvPre({T, D}, 0.0f);
    for (size_t i = 0; i < dConvPre.data.size(); ++i)
        dConvPre.data[i] = dConv.data[i] * silu_grad(conv_pre.data[i]);
    // Conv1d backward (causal depthwise): out[t,d] = bias[d] + sum_j w[d*K+j]*in[t-K+1+j,d]
    Tensor dXssm({T, D}, 0.0f);
    Tensor dConvW({D, (size_t)1, K}, 0.0f);
    Tensor dConvB({D}, 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d) {
            float g = dConvPre(t, d);
            dConvB.data[d] += g;
            for (size_t j = 0; j < K; ++j) {
                long tin = (long)t - (long)(K - 1) + (long)j;
                if (tin >= 0) {
                    dXssm((size_t)tin, d) += g * conv1d_weight_.data[d * K + j];
                    dConvW.data[d * K + j] += g * x_ssm((size_t)tin, d);
                }
            }
        }
    const_cast<Tensor&>(conv1d_weight_).add_grad(dConvW);
    const_cast<Tensor&>(conv1d_bias_).add_grad(dConvB);
    // proj split: dProj[:,0:D]=dXz, dProj[:,D:]=dXssm
    Tensor dProj({T, 2 * D}, 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < D; ++d) {
            dProj(t, d) = dXz(t, d);
            dProj(t, d + D) = dXssm(t, d);
        }
    // dNormed = dProj @ in_proj ; dIn_proj[j,i] = sum_t dProj[t,j]*normed[t,i]
    Tensor dNormed({T, E}, 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < E; ++i) {
            float acc = 0.0f;
            for (size_t j = 0; j < 2 * D; ++j) acc += dProj(t, j) * in_proj_weight_(j, i);
            dNormed(t, i) = acc;
        }
    {
        Tensor dW({2 * D, E}, 0.0f);
        for (size_t j = 0; j < 2 * D; ++j)
            for (size_t i = 0; i < E; ++i) {
                float acc = 0.0f;
                for (size_t t = 0; t < T; ++t) acc += dProj(t, j) * normed(t, i);
                dW(j, i) = acc;
            }
        const_cast<Tensor&>(in_proj_weight_).add_grad(dW);
    }
    // norm backward
    Tensor dX;
    if (use_rmsnorm_) {
        auto rb = x.rmsnorm_backward(dNormed, &norm_weight_);
        dX = rb.grad_x;
        const_cast<Tensor&>(norm_weight_).add_grad(rb.grad_w);
    } else {
        auto lb = x.layernorm_backward(dNormed, &norm_weight_);
        dX = lb.grad_x;
        const_cast<Tensor&>(norm_weight_).add_grad(lb.grad_gamma);
        // layernorm beta: MambaBlock has no beta param; fold its grad away
        (void)lb.grad_beta;
    }
    // residual
    for (size_t i = 0; i < dX.data.size(); ++i) dX.data[i] += grad_out.data[i];
    return dX;
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
