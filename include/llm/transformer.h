#pragma once
#include "attention.h"
#include "feed_forward.h"
#include "kv_cache.h"

namespace llm {

struct TransformerConfig { bool pre_ln = true; float dropout=0.0f; };

class TransformerBlock {
public:
    TransformerBlock(size_t n_embd, size_t n_heads, size_t block_size, TransformerConfig cfg = {});

    Tensor forward(const Tensor& x) const;
    Tensor forward_incremental(const Tensor& x, KVCache& cache, size_t layer, size_t pos) const;
    Tensor backward(const Tensor& x, const Tensor& grad_out) const;
    // Expose for checkpoint v3 (full serialization)
    std::vector<Tensor*> parameters();
    std::vector<const Tensor*> parameters() const;

private:
    MultiHeadAttention attn_;
    FeedForward ffn_;
    Tensor ln1_gamma_, ln1_beta_;
    Tensor ln2_gamma_, ln2_beta_;
};

} // namespace llm
