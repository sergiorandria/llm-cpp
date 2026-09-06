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
    assert(res.mmlu == 0 && res.hellaswag == 0);  // stub
    // Empty data should give ppl 0
    auto res2 = llm::evaluate(model, {});
    assert(res2.ppl == 0);
    std::cout << "eval test passed\n";
    return 0;
}
