#pragma once
#include "model.h"
namespace llm {
// Speculative decoding: draft model (small, 2 layers) proposes k tokens, target model verifies in parallel via single forward pass
// 2× speedup for temp=0, k=4 — draft 10M vs target 124M — batched verification (one target forward per round)
class SpeculativeDecoder {
public:
    SpeculativeDecoder(const GPT& draft, const GPT& target, size_t k=4);
    // Greedy (temp=0) speculative: draft proposes k tokens, target verifies via single forward over out+draft
    // Returns prompt + max_tokens new tokens, token-for-token identical to target.generate(prompt, max_tokens, 0,0) when temp=0
    std::vector<int> generate(const std::vector<int>& prompt, size_t max_tokens) const;
private:
    const GPT& draft_, target_;
    size_t k_;
};
}
