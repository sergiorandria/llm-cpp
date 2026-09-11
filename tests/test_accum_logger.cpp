#include "llm/trainer.h"
#include "llm/logging.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
int main() {
    llm::Config cfg;
    cfg.vocab_size = 16; cfg.n_layers = 1; cfg.n_heads = 2; cfg.n_embd = 8; cfg.block_size = 8;
    cfg.deterministic = true;
    std::vector<int> batch = {1,2,3,4,5,6,7,8};
    // D35: accum K=4 steps only on 4th call; params unchanged before
    llm::GPT m(cfg);
    llm::AdamW o(1e-3f); llm::CosineScheduler s(1e-3f, 10, 100);
    llm::TrainConfig t; t.deterministic = true; t.grad_accum_steps = 4;
    llm::Trainer tr(m, t, o, s);
    auto before = m.parameters()[0]->data;
    tr.train_step_accum(batch);
    tr.train_step_accum(batch);
    assert(m.parameters()[0]->data == before);  // no step yet
    tr.train_step_accum(batch);
    tr.train_step_accum(batch);  // steps here
    assert(m.parameters()[0]->data != before);
    // D36: mean reduce == manual average, 1-shard identity
    llm::Tensor g1({2}, 0.0f); g1.data = {2.0f, 4.0f};
    llm::Tensor g2({2}, 0.0f); g2.data = {4.0f, 8.0f};
    std::vector<llm::Tensor> out;
    llm::mean_reduce_grads({{g1}, {g2}}, out);
    assert(std::fabs(out[0].data[0] - 3.0f) < 1e-6 && std::fabs(out[0].data[1] - 6.0f) < 1e-6);
    llm::mean_reduce_grads({{g1}}, out);
    assert(out[0].data == g1.data);
    // D37: 3 steps emit 3 parseable JSONL lines
    std::remove("/tmp/train.log.jsonl");
    llm::MetricsLogger lg("/tmp/train.log.jsonl");
    for (int i = 0; i < 3; ++i) lg.log_step(i, 2.5f - i * 0.1f, 10.0f, 1e-3f, 0.5f, 1000.0f);
    std::ifstream in("/tmp/train.log.jsonl");
    int lines = 0;
    std::string l;
    while (std::getline(in, l)) {
        assert(l.find("\"step\"") != std::string::npos && l.find("\"loss\"") != std::string::npos);
        ++lines;
    }
    assert(lines == 3);
    std::cout << "accum+reduce+logger test passed\n";
    return 0;
}
