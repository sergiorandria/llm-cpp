#pragma once
#include "tensor.h"

namespace llm {

class MultiHeadAttention {
public:
    MultiHeadAttention(size_t n_embd, size_t n_heads, size_t block_size, bool bias = false);

    // x: [seq_len, n_embd] -> out: [seq_len, n_embd]
    Tensor forward(const Tensor& x, bool causal=true, float dropout_p=0.0f) const;

    size_t n_heads_;
    size_t n_embd_;
    size_t head_dim_;
private:
    Tensor Wq_, Wk_, Wv_, Wo_;
    Tensor bq_, bk_, bv_, bo_;
};

} // namespace llm
