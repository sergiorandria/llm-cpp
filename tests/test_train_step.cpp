#include "llm/model.h"
#include "llm/trainer.h"
#include "llm/optimizer.h"
#include "llm/scheduler.h"
#include "llm/dataset.h"
#include <cassert>
#include <iostream>

int main() {
    llm::Config cfg; cfg.vocab_size=32; cfg.n_embd=16; cfg.n_heads=4; cfg.n_layers=1; cfg.block_size=16;
    llm::GPT model(cfg);
    auto params_before = model.parameters();
    std::vector<float> before;
    for (auto *p: params_before) before.insert(before.end(), p->data.begin(), p->data.end());

    llm::Dataset ds("/tmp/nonexistent.txt", 16); // fallback tokens
    llm::AdamW optim(1e-3f);
    llm::CosineScheduler sched(1e-3f, 10, 100);
    llm::TrainConfig tcfg; tcfg.max_iters=1;
    llm::Trainer trainer(model, tcfg, optim, sched);
    auto batch = ds.tokens();
    if (batch.size() > 16) batch.resize(16);
    float loss1 = trainer.train_step(batch);
    auto params_after = model.parameters();
    float diff=0; size_t idx=0;
    for (auto *p: params_after) for (float v: p->data) { diff += std::abs(v - before[idx++]); }
    std::cout << "loss " << loss1 << " param L1 diff " << diff << "\n";
    assert(diff > 1e-6 && "trainer did not update weights — optim.step not called");
    // Second step should not be NaN
    float loss2 = trainer.train_step(batch);
    assert(!std::isnan(loss2));
    std::cout << "train_step test passed (weights moved, loss finite)\n";
    return 0;
}
