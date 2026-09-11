#pragma once
#include "model.h"
#include <vector>
namespace llm {
// Real beam search: tracks cumulative log-prob per beam, expands each beam's top-k continuations,
// keeps beam_width highest-scoring sequences each step, returns best-scoring completed beam.
// E49: length penalty (score /= len^penalty) + early stopping when best completed beam
// cannot be beaten by any active beam's optimistic extension.
struct BeamConfig {
    size_t beam_width = 4;
    float len_penalty = 0.0f; // 0 = none; 0.6-1.0 typical (GNMT)
    bool early_stop = false;
};
std::vector<int> beam_search(const GPT& model, const std::vector<int>& prompt, size_t max_new_tokens, size_t beam_width=4);
std::vector<int> beam_search_cfg(const GPT& model, const std::vector<int>& prompt, size_t max_new_tokens, BeamConfig cfg);
}
