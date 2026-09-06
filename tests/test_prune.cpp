#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/prune.h"

int main() {
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_embd = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 1;
    cfg.block_size = 8;
    llm::GPT model(cfg);
    float before = llm::sparsity(model);
    std::cout << "sparsity before " << before << "\n";
    assert(before < 0.1f);
    llm::prune_model(model, 0.5f);
    float after = llm::sparsity(model);
    std::cout << "sparsity after 0.5 " << after << "\n";
    // Should be roughly 0.5 (within 0.1 due to threshold approx)
    assert(after > 0.3f && after < 0.7f);
    // Prune 0 should not change much
    llm::GPT model2(cfg);
    llm::prune_model(model2, 0.0f);
    assert(std::abs(llm::sparsity(model2) - before) < 1e-6f);
    std::cout << "prune test passed\n";
    return 0;
}
