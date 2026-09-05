#pragma once
#include "model.h"
namespace llm {
// RLHF: reward model + PPO (stub) — reward from human preference, KL penalty vs SFT
struct RewardConfig { float kl_coef=0.1f; };
class RewardModel {
public:
    explicit RewardModel(const GPT& base);
    float score(const std::vector<int>& prompt, const std::vector<int>& completion) const;
};
void ppo_step(GPT& policy, const RewardModel& reward, const std::vector<int>& prompt);
}
