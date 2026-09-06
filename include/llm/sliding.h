#pragma once
#include "tensor.h"
namespace llm {
// Sliding window attention: 512 window, 32k context via dilated windows
Tensor sliding_attention(const Tensor& Q, const Tensor& K, const Tensor& V, size_t window=512);
}
