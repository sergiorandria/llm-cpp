#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/model.h"
#include "llm/trainer.h"
int main() {
    // C29: checkpointing flag on/off must give identical grads (recompute-always design)
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_embd = 8;
    cfg.block_size = 8;
    llm::GPT m1(cfg), m2(cfg);
    auto p1 = m1.parameters();
    auto p2 = m2.parameters();
    for (size_t i = 0; i < p1.size(); ++i) p2[i]->data = p1[i]->data;
    llm::AdamW o1(1e-3f), o2(1e-3f);
    llm::CosineScheduler s1(1e-3f, 10, 100), s2(1e-3f, 10, 100);
    llm::TrainConfig t1;
    t1.checkpointing = false;
    llm::TrainConfig t2;
    t2.checkpointing = true;
    llm::Trainer tr1(m1, t1, o1, s1), tr2(m2, t2, o2, s2);
    std::vector<int> batch = {1, 2, 3, 4, 5, 6, 7, 8};
    tr1.train_step(batch);
    tr2.train_step(batch);
    auto q1 = m1.parameters();
    auto q2 = m2.parameters();
    float md = 0;
    for (size_t i = 0; i < q1.size(); ++i)
        for (size_t j = 0; j < q1[i]->data.size(); ++j)
            md = std::max(md, std::fabs(q1[i]->data[j] - q2[i]->data[j]));
    std::cout << "ckpt on/off maxd=" << md << "\n";
    assert(md < 1e-5);
    // C30: extend 8 -> 16 keeps prefix rows, 16-token forward finite
    llm::GPT ms(cfg);
    ms.extend_context(16);
    assert(ms.config().block_size == 16);
    std::vector<int> long_batch(16);
    for (int i = 0; i < 16; ++i) long_batch[i] = i;
    auto logits = ms.forward(long_batch);
    assert(logits.shape[0] == 16);
    for (auto v : logits.data) assert(std::isfinite(v));
    ms.extend_context(8);  // shrink is no-op
    assert(ms.config().block_size == 16);
    std::cout << "ckpt+extend test passed\n";
    return 0;
}
