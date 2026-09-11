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
// F54: quantized storage. qtype: 0 = F32 (== save_gguf), 2 = Q4_0, 8 = Q8_0
// (ggml type ids; block 32 with F32 super-scale). Load auto-detects per tensor
// and dequantizes into F32 model weights.
bool save_gguf_quant(const GPT& model, const std::string& path, int qtype);
// Block codecs (exposed for tests)
void encode_q80_block(const float* x, float& scale_out, int8_t* q_out);  // 32 vals
void decode_q80_block(float scale, const int8_t* q, float* out);
void encode_q40_block(const float* x, float& scale_out, uint8_t* packed_out);  // 32 vals -> 16B
void decode_q40_block(float scale, const uint8_t* packed, float* out);
}  // namespace llm
