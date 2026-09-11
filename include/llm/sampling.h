#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
namespace llm {
int sample_greedy(const std::vector<float>& logits);
int sample_temperature(const std::vector<float>& logits, float temp);
int sample_top_k(const std::vector<float>& logits, int k, float temp = 1.0f);
int sample_top_p(const std::vector<float>& logits, float p, float temp = 1.0f);
std::vector<float> apply_repetition_penalty(const std::vector<float>& logits,
                                            const std::vector<int>& generated, float penalty);
// E46: OpenAI-style frequency/presence penalties
std::vector<float> apply_freq_presence(const std::vector<float>& logits,
                                       const std::vector<int>& generated, float freq_penalty,
                                       float pres_penalty);
// E47: per-token logit bias (-inf effectively bans); applied before temperature
std::vector<float> apply_logit_bias(const std::vector<float>& logits,
                                    const std::unordered_map<int, float>& bias);
// E44: stop on EOS id or any stop sequence id (single-token stops; multi-token handled by caller)
bool should_stop(int next_id, int eos_id, const std::vector<int>& stop_ids);
int sample_temperature_seeded(const std::vector<float>& logits, float temp, uint64_t seed);
int sample_top_k_seeded(const std::vector<float>& logits, int k, float temp, uint64_t seed);
int sample_top_p_seeded(const std::vector<float>& logits, float p, float temp, uint64_t seed);
}  // namespace llm
