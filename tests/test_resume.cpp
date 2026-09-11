#include "llm/trainer.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
int main() {
    llm::Config cfg;
    cfg.vocab_size = 16; cfg.n_layers = 1; cfg.n_heads = 2; cfg.n_embd = 8; cfg.block_size = 8;
    cfg.deterministic = true;
    std::vector<int> batch = {1,2,3,4,5,6,7,8};
    auto run_continuous = [&]() {
        llm::GPT m(cfg);
        llm::AdamW o(1e-3f); llm::CosineScheduler s(1e-3f, 10, 100);
        llm::TrainConfig t; t.deterministic = true;
        llm::Trainer tr(m, t, o, s);
        for (int i = 0; i < 10; ++i) { tr.train_step(batch); tr.set_step(tr.step() + 1); }
        return m;
    };
    // reference: capture init then run 10
    llm::GPT m0(cfg);
    auto p0 = m0.parameters();
    // continuous 10 from m0's init: emulate by copying init into fresh and running
    llm::GPT mref(cfg);
    for (size_t i = 0; i < p0.size(); ++i) mref.parameters()[i]->data = p0[i]->data;
    {
        llm::AdamW o(1e-3f); llm::CosineScheduler s(1e-3f, 10, 100);
        llm::TrainConfig t; t.deterministic = true;
        llm::Trainer tr(mref, t, o, s);
        for (int i = 0; i < 10; ++i) { tr.train_step(batch); tr.set_step(tr.step() + 1); }
    }
    // resume: 5 steps, save, new trainer load, 5 more
    llm::GPT mres(cfg);
    for (size_t i = 0; i < p0.size(); ++i) mres.parameters()[i]->data = p0[i]->data;
    {
        llm::AdamW o(1e-3f); llm::CosineScheduler s(1e-3f, 10, 100);
        llm::TrainConfig t; t.deterministic = true;
        llm::Trainer tr(mres, t, o, s);
        for (int i = 0; i < 5; ++i) { tr.train_step(batch); tr.set_step(tr.step() + 1); }
        tr.save_train_state("/tmp/resume_ckpt");
    }
    {
        llm::AdamW o(1e-3f); llm::CosineScheduler s(1e-3f, 10, 100);
        llm::TrainConfig t; t.deterministic = true;
        llm::Trainer tr(mres, t, o, s);
        tr.load_train_state("/tmp/resume_ckpt");
        assert(tr.step() == 5);
        for (int i = 0; i < 5; ++i) { tr.train_step(batch); tr.set_step(tr.step() + 1); }
        assert(tr.step() == 10);
    }
    auto pr = mref.parameters(), ps = mres.parameters();
    float md = 0;
    for (size_t i = 0; i < pr.size(); ++i)
        for (size_t j = 0; j < pr[i]->data.size(); ++j)
            md = std::max(md, std::fabs(pr[i]->data[j]-ps[i]->data[j]));
    std::cout << "resume maxd=" << md << "\n";
    assert(md == 0.0f);  // bit-exact in deterministic mode
    std::cout << "resume test passed\n";
    (void)run_continuous;
    return 0;
}
