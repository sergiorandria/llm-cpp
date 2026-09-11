// I89: inference benchmark — tokens/sec + trial p50/p95.
// NOTE: pin OMP_NUM_THREADS=1 for small models; nested OpenMP (ours x numpy-cpp)
// oversubscribes micro-GEMMs (~160x slower with 12 threads on 64-embd).
// Usage: ./build/bench_infer [--tokens N] [--prompt "..."] [--trials K]
#include "llm/model.h"
#include "llm/tokenizer.h"
#include "llm/utils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    size_t max_new = 20, trials = 5;
    std::string prompt = "Hello LLM benchmark";
    for (int i = 1; i + 1 < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tokens") max_new = std::stoul(argv[++i]);
        else if (a == "--prompt") prompt = argv[++i];
        else if (a == "--trials") trials = std::stoul(argv[++i]);
    }
    llm::Config cfg;
    cfg.vocab_size = 256; cfg.n_layers = 2; cfg.n_heads = 4; cfg.n_embd = 64; cfg.block_size = 128;
    llm::GPT model(cfg);
    llm::Tokenizer tok(256);
    auto ids = tok.encode(prompt);
    model.generate(ids, 2, 0.0f);  // warmup
    std::vector<double> totals;
    for (size_t t = 0; t < trials; ++t) {
        auto t0 = std::chrono::steady_clock::now();
        (void)model.generate(ids, max_new, 0.0f);
        auto t1 = std::chrono::steady_clock::now();
        totals.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(totals.begin(), totals.end());
    double p50 = totals[totals.size() / 2];
    double p95 = totals[(totals.size() * 95) / 100];
    double ms_tok = p50 / max_new;
    std::cout << "infer bench: tokens=" << max_new << " trials=" << trials
              << " total_p50_ms=" << p50 << " total_p95_ms=" << p95
              << " ms/token=" << ms_tok << " tokens/sec=" << 1000.0 / ms_tok
              << " caps=" << llm::simd_caps() << "\n";
    return 0;
}
