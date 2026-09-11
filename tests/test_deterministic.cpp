#include <cassert>
#include <iostream>

#include "llm/trainer.h"
int main() {
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_embd = 8;
    cfg.block_size = 8;
    cfg.deterministic = true;
    llm::GPT m1(cfg), m2(cfg);
    auto p1 = m1.parameters();
    auto p2 = m2.parameters();
    for (size_t i = 0; i < p1.size(); ++i) p2[i]->data = p1[i]->data;
    llm::AdamW o1(1e-3f), o2(1e-3f);
    llm::CosineScheduler s1(1e-3f, 10, 100), s2(1e-3f, 10, 100);
    llm::TrainConfig t;
    t.deterministic = true;
    llm::Trainer tr1(m1, t, o1, s1), tr2(m2, t, o2, s2);
    std::vector<int> batch = {1, 2, 3, 4, 5, 6, 7, 8};
    float l1 = tr1.train_step(batch);
    float l2 = tr2.train_step(batch);
    assert(l1 == l2);
    auto q1 = m1.parameters();
    auto q2 = m2.parameters();
    for (size_t i = 0; i < q1.size(); ++i) assert(q1[i]->data == q2[i]->data);
    std::cout << "deterministic test passed loss=" << l1 << "\n";
    return 0;
}
