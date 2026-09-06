#pragma once
#include "model.h"
namespace llm {
// RLHF: reward model + PPO — reward from human preference, KL penalty vs SFT
struct RewardConfig {
    float kl_coef = 0.1f;
    float ppo_clip_eps = 0.2f;
};
class RewardModel {
   public:
    explicit RewardModel(const GPT& base);
    // Inference: reward for prompt+completion via learned head
    float score(const std::vector<int>& prompt, const std::vector<int>& completion) const;
    // Training: Bradley-Terry pairwise logistic loss on (chosen vs rejected)
    void train_step(const std::vector<int>& prompt, const std::vector<int>& chosen,
                    const std::vector<int>& rejected, float lr = 1e-2f);
    // For testing: access to head
    const Tensor& head_weight() const { return head_w_; }
    float head_bias() const { return head_b_; }

   private:
    const GPT& base_;
    Tensor head_w_; // [n_embd] linear head
    float head_b_ = 0.0f;
    Tensor get_pooled_hidden(const std::vector<int>& prompt, const std::vector<int>& completion) const;
};
void ppo_step(GPT& policy, const RewardModel& reward, const std::vector<int>& prompt);
void ppo_step(GPT& policy, const RewardModel& reward, const std::vector<int>& prompt, const RewardConfig& cfg);
}  // namespace llm
