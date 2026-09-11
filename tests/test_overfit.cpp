// J95: overfit integration — tiny Shakespeare slice must show real learning.
// 200 lines, 2-layer 64-embd, 100 steps: loss slope < 0 and final < initial/1.5.
#include "llm/dataset.h"
#include "llm/loss.h"
#include "llm/model.h"
#include "llm/tokenizer.h"
#include "llm/trainer.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
int main() {
    std::ifstream in("data/malagasy/processed/shakespeare.txt");
    assert(in && "run from repo root (WORKING_DIRECTORY set)");
    std::vector<int> toks;
    std::string line;
    int lines = 0;
    llm::Tokenizer tok(256);
    while (std::getline(in, line) && lines < 200) {
        if (line.empty()) continue;
        ++lines;
        auto ids = tok.encode(line + "\n");
        toks.insert(toks.end(), ids.begin(), ids.end());
    }
    assert(toks.size() > 512);
    llm::Config cfg;
    cfg.vocab_size = 256; cfg.n_layers = 2; cfg.n_heads = 4; cfg.n_embd = 64; cfg.block_size = 32;
    llm::GPT m(cfg);
    llm::AdamW o(3e-3f);
    llm::CosineScheduler s(3e-3f, 10, 1000);
    llm::TrainConfig t;
    t.deterministic = true;
    llm::Trainer tr(m, t, o, s);
    const size_t W = 32;
    std::vector<float> losses;
    for (int step = 0; step < 100; ++step) {
        size_t off = (step * W) % (toks.size() - W);
        std::vector<int> batch(toks.begin() + off, toks.begin() + off + W);
        losses.push_back(tr.train_step(batch));
        tr.set_step(tr.step() + 1);
    }
    // linear-fit slope of loss must be negative (overall downward)
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < losses.size(); ++i) {
        sx += (double)i; sy += losses[i]; sxx += (double)i * i; sxy += (double)i * losses[i];
    }
    double n = losses.size();
    double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    std::cout << "overfit: first=" << losses.front() << " last=" << losses.back()
              << " slope=" << slope << "\n";
    assert(slope < 0);
    assert(losses.back() < losses.front() / 1.5f);
    assert(tr.skipped_steps() == 0);
    std::cout << "overfit test passed\n";
    return 0;
}
