#pragma once
#include <vector>
namespace llm {
int sample_greedy(const std::vector<float>& logits);
int sample_temperature(const std::vector<float>& logits, float temp);
int sample_top_k(const std::vector<float>& logits, int k, float temp=1.0f);
int sample_top_p(const std::vector<float>& logits, float p, float temp=1.0f);
std::vector<float> apply_repetition_penalty(const std::vector<float>& logits, const std::vector<int>& generated, float penalty);
}
