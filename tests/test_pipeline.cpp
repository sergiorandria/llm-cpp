#include <cassert>
#include <iostream>

#include "llm/pipeline.h"

int main() {
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_embd = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 2;
    cfg.block_size = 16;
    llm::GPT model(cfg);
    llm::Pipeline pipe(model, 3, 2);
    std::vector<int> batch = {1, 2, 3, 4, 5, 6, 7, 8};
    auto before = model.parameters();
    std::vector<float> bv;
    for (auto* p : before) bv.insert(bv.end(), p->data.begin(), p->data.end());
    pipe.train_step(batch);
    auto after = model.parameters();
    float diff = 0;
    size_t idx = 0;
    for (auto* p : after)
        for (float v : p->data) diff += std::abs(v - bv[idx++]);
    std::cout << "pipeline param diff " << diff << "\n";
    assert(diff > 1e-8f);
    // Empty batch should not crash
    std::vector<int> empty;
    pipe.train_step(empty);
    std::cout << "pipeline test passed\n";
    return 0;
}
