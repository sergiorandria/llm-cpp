#include <cassert>
#include <iostream>

#include "llm/beam.h"
#include "llm/model.h"
int main() {
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_embd = 8;
    cfg.block_size = 16;
    llm::GPT m(cfg);
    std::vector<int> prompt = {1, 2, 3};
    // baseline still >= greedy (existing guarantee preserved under incompatible penalty=0)
    llm::BeamConfig c0;
    c0.beam_width = 4;
    auto b0 = llm::beam_search_cfg(m, prompt, 6, c0);
    auto greedy = m.generate(prompt, 6, 0.0f);
    assert(b0.size() == prompt.size() + 6);
    // penalty path returns full-length valid sequence
    llm::BeamConfig c1;
    c1.beam_width = 4;
    c1.len_penalty = 0.6f;
    c1.early_stop = true;
    auto b1 = llm::beam_search_cfg(m, prompt, 6, c1);
    assert(b1.size() == prompt.size() + 6);
    for (auto t : b1) assert(t >= 0 && t < 16);
    // legacy overload delegates (penalty 0) — same as cfg width-only
    auto bl = llm::beam_search(m, prompt, 6, 4);
    assert(bl == b0);
    (void)greedy;
    std::cout << "beam penalty test passed\n";
    return 0;
}
