#pragma once
#include "tensor.h"

namespace llm {

class MultiHeadAttention {
public:
    MultiHeadAttention(size_t n_embd, size_t n_heads, size_t block_size, bool bias = false);

    // x: [seq_len, n_embd] -> out: [seq_len, n_embd]
    Tensor forward(const Tensor& x, bool causal=true, float dropout_p=0.0f) const;
    // Manual backward: given x and grad_out [T,C], returns grad_x and accumulates param grads
    Tensor backward(const Tensor& x, const Tensor& grad_out) const;

    size_t n_heads_;
    size_t n_embd_;
    size_t head_dim_;
    bool bias_;
    // KV-cache for inference — currently unused in GPT::generate (O(n²) recompute).
    // Canonical cache is llm::KVCache (kv_cache.h); this per-attention vector is deprecated
    // and will be removed in favor of KVCache passed through forward().
    // See docs/NUMPY_BACKEND.md and GitHub issue: generation still recomputes full context.
    mutable std::vector<Tensor> k_cache_, v_cache_;
    void clear_cache() const { k_cache_.clear(); v_cache_.clear(); }
    std::vector<Tensor*> parameters();
    std::vector<const Tensor*> parameters() const;
private:
    Tensor Wq_, Wk_, Wv_, Wo_;
    Tensor bq_, bk_, bv_, bo_;
};

} // namespace llm
