#pragma once
#include "tensor.h"
namespace llm {
// FlashAttention-style tiled attention: O(n) memory, block-wise softmax
// For head_dim up to 128, tile size 64 balances cache locality
Tensor flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                       float scale, bool causal, size_t block_size=64);
}
