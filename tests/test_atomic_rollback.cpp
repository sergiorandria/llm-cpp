#include "llm/model.h"
#include "llm/trainer.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
int main() {
    llm::Config cfg;
    cfg.vocab_size = 16; cfg.n_layers = 1; cfg.n_heads = 2; cfg.n_embd = 8; cfg.block_size = 8;
    // H78: atomic save over existing path leaves valid loadable file, no .tmp
    llm::GPT m1(cfg);
    m1.save_binary_atomic("/tmp/atomic.bin");
    for (auto p : m1.parameters()) for (auto& v : p->data) v += 1.0f;
    m1.save_binary_atomic("/tmp/atomic.bin");  // overwrite atomically
    bool no_tmp = !std::ifstream("/tmp/atomic.bin.tmp").good();
    assert(no_tmp);
    llm::GPT m2(cfg);
    for (auto p : m2.parameters()) for (auto& v : p->data) v -= 5.0f;
    m2.load_binary("/tmp/atomic.bin");
    float md = 0;
    auto p1 = m1.parameters(); auto p2 = m2.parameters();
    for (size_t i = 0; i < p1.size(); ++i)
        for (size_t j = 0; j < p1[i]->data.size(); ++j)
            md = std::max(md, std::fabs(p1[i]->data[j] - p2[i]->data[j]));
    assert(md == 0.0f);
    // H79: rollback restores snapshot + halves LR; finite loss is no-op
    llm::GPT m3(cfg);
    llm::AdamW o(0.01f);
    llm::CosineScheduler s(0.01f, 10, 100);
    llm::TrainConfig t;
    llm::Trainer tr(m3, t, o, s);
    tr.snapshot_params();
    float lr_before = o.get_lr();
    bool rb0 = tr.rollback_if_nonfinite(2.5f);
    assert(!rb0 && o.get_lr() == lr_before);
    m3.parameters()[0]->data[0] = std::numeric_limits<float>::quiet_NaN();
    bool rb1 = tr.rollback_if_nonfinite(std::numeric_limits<float>::quiet_NaN());
    assert(rb1);
    assert(std::isfinite(m3.parameters()[0]->data[0]));
    assert(std::fabs(o.get_lr() - lr_before * 0.5f) < 1e-9);
    std::cout << "atomic+rollback test passed\n";
    return 0;
}
