#pragma once
#include <string>

#include "model.h"
namespace llm {
// GGUF I/O — real binary format (header magic + KV metadata + tensor info + data)
// Magic 0x46554747 ("GGUF"), version 3, 32-byte aligned data section.
// Saves all GPT::parameters() with names tensor_0.. and shape/dtype/offset.
// Load validates magic/version/config and populates tensors bit-exact.
bool save_gguf(const GPT& model, const std::string& path);
bool load_gguf(GPT& model, const std::string& path);
// F54: quantized storage. qtype: 0 = F32 (== save_gguf), 2 = Q4_0, 8 = Q8_0,
// 12 = Q4_K super-blocks (ggml type ids; Q4_0/Q8_0 use block 32 with F32
// super-scale, Q4_K uses 256-elem super-blocks, see below). Load
// auto-detects per tensor and dequantizes into F32 model weights.
bool save_gguf_quant(const GPT& model, const std::string& path, int qtype);
// Block codecs (exposed for tests)
void encode_q80_block(const float* x, float& scale_out, int8_t* q_out);  // 32 vals
void decode_q80_block(float scale, const int8_t* q, float* out);
void encode_q40_block(const float* x, float& scale_out, uint8_t* packed_out);  // 32 vals -> 16B
void decode_q40_block(float scale, const uint8_t* packed, float* out);
// Q4_K super-block codec (ggml block_q4_K layout, 144 bytes per 256 vals):
// d/dmin fp16 super-scales, scales[12] (8x6-bit sub-scales + 8x6-bit
// sub-mins, same packing as ggml get_scale_min_k4), qs[128] (256x4-bit).
// Dequant: y = dall*sc*q - dmin*m. Mins anchor: dmin covers max(-mn) when
// some sub-min is negative, else -max(mn) (exact for single-sign-min
// blocks, i.e. all realistic weight tensors); wildly mixed-sign sub-mins
// within one super-block saturate the minority side (bounded, clamped).
void encode_q4k_block(const float* x, uint8_t* out144);  // 256 vals -> 144B
void decode_q4k_block(const uint8_t* in144, float* out);   // 144B -> 256 vals
inline constexpr size_t GGUF_Q4K = 256;
inline constexpr size_t GGUF_Q4K_BYTES = 144;
}  // namespace llm
