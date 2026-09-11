#include "llm/trainer.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
int main() {
    // A07 helper
    llm::Tensor g({2}, 0.0f); g.data = {1.0f, 2.0f};
    llm::Tensor bad({2}, 0.0f); bad.data = {1.0f, std::numeric_limits<float>::quiet_NaN()};
    llm::Tensor inf({1}, 0.0f); inf.data = {std::numeric_limits<float>::infinity()};
    assert(!llm::has_nonfinite({g}));
    assert(llm::has_nonfinite({g, bad}));
    assert(llm::has_nonfinite({inf}));

    // A09: loss_scale=1024 path equals scale=1 within tolerance (linear unscale)
    llm::Config cfg;
    cfg.vocab_size = 16; cfg.n_layers = 1; cfg.n_heads = 2; cfg.n_embd = 8; cfg.block_size = 8;
    llm::GPT m1(cfg), m2(cfg);
    // copy params m1->m2 for identical start
    auto p1 = m1.parameters(); auto p2 = m2.parameters();
    for (size_t i = 0; i < p1.size(); ++i) p2[i]->data = p1[i]->data;
    llm::AdamW o1(1e-3f), o2(1e-3f);
    llm::CosineScheduler s1(1e-3f, 10, 100), s2(1e-3f, 10, 100);
    llm::TrainConfig t1; t1.loss_scale = 1.0f;
    llm::TrainConfig t2; t2.loss_scale = 1024.0f;
    llm::Trainer tr1(m1, t1, o1, s1), tr2(m2, t2, o2, s2);
    std::vector<int> batch = {1, 2, 3, 4, 5, 6, 7, 8};
    tr1.train_step(batch);
    tr2.train_step(batch);
    assert(tr1.skipped_steps() == 0 && tr2.skipped_steps() == 0);
    auto q1 = m1.parameters(); auto q2 = m2.parameters();
    float maxd = 0;
    for (size_t i = 0; i < q1.size(); ++i)
        for (size_t j = 0; j < q1[i]->data.size(); ++j)
            maxd = std::max(maxd, std::fabs(q1[i]->data[j] - q2[i]->data[j]));
    std::cout << "loss_scale maxd=" << maxd << "\n";
    assert(maxd < 1e-3);
    // normal step not skipped
    assert(tr1.step() == 0);  // step_ only advances in train(), train_step alone doesn't bump
    std::cout << "nan_guard + loss_scale test passed\n";
    return 0;
}
