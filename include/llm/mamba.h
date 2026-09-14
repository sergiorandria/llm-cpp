// include/llm/mamba.h
#pragma once
#include <vector>
#include "tensor.h"

namespace llm {

class MambaBlock {
public:
    MambaBlock(size_t n_embd, size_t d_inner, size_t d_state,
               size_t dt_rank, size_t conv_kernel, bool use_rmsnorm = true);

    Tensor forward(const Tensor& x) const;
    Tensor backward(const Tensor& x, const Tensor& grad_out) const;

    std::vector<Tensor*> parameters();
    std::vector<const Tensor*> parameters() const;

    size_t n_embd() const { return n_embd_; }
    size_t d_inner() const { return d_inner_; }
    size_t d_state() const { return d_state_; }

private:
    size_t n_embd_, d_inner_, d_state_, dt_rank_, conv_kernel_;
    bool use_rmsnorm_;

    // Parameters
    Tensor norm_weight_;                  // [n_embd] RMSNorm
    Tensor in_proj_weight_;               // [2*d_inner, n_embd]
    Tensor conv1d_weight_;                // [d_inner, 1, conv_kernel]
    Tensor conv1d_bias_;                  // [d_inner]
    Tensor x_proj_weight_;                // [dt_rank + 2*d_state, d_inner]
    Tensor dt_proj_weight_;               // [d_inner, dt_rank]
    Tensor a_log_;                        // [d_inner, d_state]
    Tensor d_weight_;                     // [d_inner]
    Tensor out_proj_weight_;              // [n_embd, d_inner]

    // Conv1D state (for incremental inference)
    mutable std::vector<float> conv_state_;  // [d_inner * (conv_kernel-1)]

    // Helper: depth-wise conv1d
    Tensor conv1d_forward(const Tensor& x) const;
};

}  // namespace llm
