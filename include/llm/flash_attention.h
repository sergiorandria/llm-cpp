#pragma once
#include "tensor.h"
namespace llm {
// FlashAttention-style tiled attention: O(n) memory, block-wise softmax
// For head_dim up to 128, tile size 64 balances cache locality
Tensor flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                       float scale, bool causal, size_t block_size=64);
// Incremental flash: Q 1×D, K/V K_len×D single query tiled
Tensor flash_attention_incremental(const Tensor& Q, const Tensor& K, const Tensor& V,
                                   float scale, size_t block_size=64);
// I84 FlashAttention-2 full prefill: online softmax over K-blocks — O(D+block)
// aux memory (no [T] exp buffer), single rescaling pass per query row.
Tensor flash_attention_full(const Tensor& Q, const Tensor& K, const Tensor& V,
                            float scale, bool causal, size_t block_size=64);
}
