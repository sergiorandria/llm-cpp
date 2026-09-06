#pragma once
#include "model.h"
#include <vector>
namespace llm {
// Real beam search: tracks cumulative log-prob per beam, expands each beam's top-k continuations,
// keeps beam_width highest-scoring sequences each step, returns best-scoring completed beam.
std::vector<int> beam_search(const GPT& model, const std::vector<int>& prompt, size_t max_new_tokens, size_t beam_width=4);
}
