#pragma once
#include "model.h"
#include <string>
namespace llm {
bool save_gguf(const GPT& model, const std::string& path);
bool load_gguf(GPT& model, const std::string& path);
}
