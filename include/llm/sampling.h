#pragma once
#include <cstdint>
#include <vector>
namespace llm {
int sample_greedy(const std::vector<float>& logits);
int sample_temperature(const std::vector<float>& logits, float temp);
int sample_top_k(const std::vector<float>& logits, int k, float temp=1.0f);
int sample_top_p(const std::vector<float>& logits, float p, float temp=1.0f);
std::vector<float> apply_repetition_penalty(const std::vector<float>& logits, const std::vector<int>& generated, float penalty);
// A06: seeded variants — same seed+logits => same token (deterministic sampling).
int sample_temperature_seeded(const std::vector<float>& logits, float temp, uint64_t seed);
int sample_top_k_seeded(const std::vector<float>& logits, int k, float temp, uint64_t seed);
int sample_top_p_seeded(const std::vector<float>& logits, float p, float temp, uint64_t seed);
}
