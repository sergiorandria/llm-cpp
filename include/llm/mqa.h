#pragma once
#include "tensor.h"
namespace llm {
// MQA: single KV head shared across Q heads — 8× KV-cache reduction vs MHA
Tensor mqa_forward(const Tensor& x, const Tensor& Wq, const Tensor& Wkv, size_t n_heads,
                   size_t head_dim);
// C21 GQA: n_kv_heads KV heads shared over n_heads Q heads (n_heads % n_kv_heads == 0).
// n_kv==n_heads ≡ MHA, n_kv==1 ≡ MQA (with split Wk/Wv layout).
Tensor gqa_forward(const Tensor& x, const Tensor& Wq, const Tensor& Wk, const Tensor& Wv,
                   size_t n_heads, size_t n_kv_heads, size_t head_dim);
}  // namespace llm
