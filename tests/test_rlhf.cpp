#include <cassert>
#include <iostream>

#include "llm/rlhf.h"

int main() {
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_embd = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 1;
    cfg.block_size = 16;
    llm::GPT base(cfg);
    llm::RewardModel rm(base);
    std::vector<int> prompt = {1, 2, 3};
    std::vector<int> comp1 = {4, 5};
    std::vector<int> comp2 = {0, 0};
    float s1 = rm.score(prompt, comp1);
    float s2 = rm.score(prompt, comp2);
    std::cout << "reward s1 " << s1 << " s2 " << s2 << "\n";
    assert(!std::isnan(s1) && !std::isnan(s2));
    assert(!std::isinf(s1) && !std::isinf(s2));
    // Scores should be deterministic
    float s1_again = rm.score(prompt, comp1);
    assert(std::abs(s1 - s1_again) < 1e-6f);

    llm::GPT policy(cfg);
    auto before = policy.parameters();
    std::vector<float> bv;
    for (auto* p : before) bv.insert(bv.end(), p->data.begin(), p->data.end());
    llm::ppo_step(policy, rm, prompt);
    auto after = policy.parameters();
    float diff = 0;
    size_t idx = 0;
    for (auto* p : after)
        for (float v : p->data) diff += std::abs(v - bv[idx++]);
    std::cout << "ppo param diff " << diff << "\n";
    assert(diff > 1e-8f);
    std::cout << "rlhf test passed\n";
    return 0;
}
