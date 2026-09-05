#pragma once
#include "tensor.h"
namespace llm {
// ALiBi: linear bias per head, extrapolates to 8k without RoPE retrain
Tensor alibi_bias(size_t seq_len, size_t n_heads);
}
