#pragma once
#include "model.h"
#include <string>
namespace llm {
// GGUF I/O — real binary format (header magic + KV metadata + tensor info + data)
// Magic 0x46554747 ("GGUF"), version 3, 32-byte aligned data section.
// Saves all GPT::parameters() with names tensor_0.. and shape/dtype/offset.
// Load validates magic/version/config and populates tensors bit-exact.
bool save_gguf(const GPT& model, const std::string& path);
bool load_gguf(GPT& model, const std::string& path);
}
