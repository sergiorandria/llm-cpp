#include "llm/model.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
    llm::Config cfg;
    cfg.vocab_size = 16; cfg.n_layers = 1; cfg.n_heads = 2; cfg.n_embd = 8; cfg.block_size = 16;
    llm::GPT m(cfg);
    std::vector<std::vector<int>> prompts = {{1,2,3}, {4,5}, {6,7,8,9,10}, {}};
    // E41: batch greedy == singles
    auto bo = m.generate_batch(prompts, 6, 0.0f);
    assert(bo.size() == prompts.size());
    for (size_t i = 0; i < prompts.size(); ++i) {
        auto s = m.generate(prompts[i], 6, 0.0f);
        assert(bo[i] == s);
    }
    // E43: streaming cb sequence == suffix, == greedy generate
    std::vector<int> streamed;
    auto out = m.generate_streaming({1,2,3}, 6, [&](int t){ streamed.push_back(t); });
    auto greedy = m.generate({1,2,3}, 6, 0.0f);
    assert(out == greedy);
    assert(streamed == std::vector<int>(out.begin() + 3, out.end()));
    // E45: logprobs valid (<=0), greedy token has max logprob, matches generate
    auto go = m.generate_with_logprobs({1,2,3}, 6, 0.0f);
    assert(go.tokens == greedy);
    assert(go.logprobs.size() == 6);
    for (auto lp : go.logprobs) assert(lp <= 0.0f && std::isfinite(lp));
    std::cout << "infer batch/stream/logprobs test passed\n";
    return 0;
}
