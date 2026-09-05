#pragma once
#include "tensor.h"
namespace llm {
// MQA: single KV head shared across Q heads — 8× KV-cache reduction vs MHA
Tensor mqa_forward(const Tensor& x, const Tensor& Wq, const Tensor& Wkv, size_t n_heads, size_t head_dim);
}
