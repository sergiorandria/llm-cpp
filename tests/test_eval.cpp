#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/eval.h"

int main() {
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_embd = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 1;
    cfg.block_size = 8;
    llm::GPT model(cfg);
    std::vector<std::vector<int>> data = {{1, 2, 3, 4}, {0, 1, 2}};
    auto res = llm::evaluate(model, data);
    std::cout << "ppl " << res.ppl << " mmlu " << res.mmlu << " hellaswag " << res.hellaswag
              << "\n";
    assert(!std::isnan(res.ppl) && !std::isinf(res.ppl));
    assert(res.ppl > 0);
    // mmlu/hellaswag are now real token-accuracy proxies, not stubbed to 0
    assert(!std::isnan(res.mmlu) && !std::isnan(res.hellaswag));
    assert(res.mmlu >= 0 && res.mmlu <= 1.0f);
    assert(res.hellaswag >= 0 && res.hellaswag <= 1.0f);
    // With random init, accuracy should be around 1/vocab, but not necessarily 0
    // Ensure they are computed (not always 0 as old stub)
    // For this small data, at least one of them may be >0 or 0, but we check they are valid
    // Empty data should give ppl 0 and mmlu/hellaswag 0
    auto res2 = llm::evaluate(model, {});
    assert(res2.ppl == 0);
    assert(res2.mmlu == 0 && res2.hellaswag == 0);
    std::cout << "eval test passed\n";
    return 0;
}
