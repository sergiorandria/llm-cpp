#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/model.h"
#include "llm/trainer.h"
int main() {
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_embd = 8;
    cfg.block_size = 8;
    cfg.use_rmsnorm = true;
    llm::GPT m(cfg);
    std::vector<int> batch = {1, 2, 3, 4, 5, 6, 7, 8};
    auto logits = m.forward(batch);
    for (auto v : logits.data) assert(std::isfinite(v));
    // 1 train step works with RMS backward
    llm::AdamW o(1e-3f);
    llm::CosineScheduler s(1e-3f, 10, 100);
    llm::TrainConfig t;
    llm::Trainer tr(m, t, o, s);
    float loss = tr.train_step(batch);
    assert(std::isfinite(loss) && loss > 0);
    assert(tr.skipped_steps() == 0);
    // incremental path also uses RMS
    auto gen = m.generate({1, 2}, 4, 0.0f);
    assert(gen.size() == 6);
    std::cout << "rmsnorm model test passed loss=" << loss << "\n";
    return 0;
}
