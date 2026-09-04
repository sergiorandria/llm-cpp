#pragma once
#include "model.h"
#include <vector>
namespace llm {
std::vector<int> beam_search(const GPT& model, const std::vector<int>& prompt, size_t max_new_tokens, size_t beam_width=4);
}
