#pragma once
#include "model.h"
namespace llm {
// Speculative decoding: draft model (small, 2 layers) proposes k tokens, target model verifies in parallel
// 2× speedup for temp=0, k=4 — draft 10M vs target 124M
class SpeculativeDecoder {
public:
    SpeculativeDecoder(const GPT& draft, const GPT& target, size_t k=4);
    std::vector<int> generate(const std::vector<int>& prompt, size_t max_tokens) const;
private:
    const GPT& draft_, target_;
    size_t k_;
};
}
