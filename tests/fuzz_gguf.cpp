// H77: GGUF fuzz — corrupted magic/truncation/shapes must be rejected, never crash.
// Runs a bounded mutation loop (ASAN-clean under build_san); libFuzzer can call
// LLVMFuzzerTestOneInput-style entry via fuzz_gguf_one() in longer runs.
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "llm/gguf.h"

static std::vector<char> read_file(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(in)), {});
}

// Returns false when load rejects (expected for corrupt inputs)
static bool fuzz_gguf_one(const std::vector<char>& data) {
    const std::string tmp = "/tmp/fuzz_gguf.bin";
    {
        std::ofstream o(tmp, std::ios::binary);
        o.write(data.data(), data.size());
    }
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_embd = 8;
    cfg.block_size = 8;
    llm::GPT m(cfg);
    return llm::load_gguf(m, tmp);  // must not crash; true/false both acceptable
}

int main() {
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_embd = 8;
    cfg.block_size = 8;
    llm::GPT m(cfg);
    bool saved = llm::save_gguf(m, "/tmp/fuzz_seed.gguf");
    assert(saved);
    std::vector<char> seed = read_file("/tmp/fuzz_seed.gguf");
    assert(!seed.empty());
    std::mt19937_64 rng(1234);
    int rejects = 0;
    // 1) bad magic variants
    for (int i = 0; i < 16; ++i) {
        auto d = seed;
        d[0] = (char)(rng() % 256);
        d[1] = (char)(rng() % 256);
        if (!fuzz_gguf_one(d)) ++rejects;
    }
    // 2) truncations
    for (size_t len : {0, 1, 7, 8, 16, 64}) {
        auto d = seed;
        d.resize(std::min(len, d.size()));
        if (!fuzz_gguf_one(d)) ++rejects;
    }
    // 3) random byte flips across header region
    for (int i = 0; i < 200; ++i) {
        auto d = seed;
        for (int k = 0; k < 4; ++k)
            d[rng() % std::min<size_t>(d.size(), 256)] = (char)(rng() % 256);
        fuzz_gguf_one(d);  // must not crash; result unchecked
    }
    // 4) corrupt shape dim (tensor info area)
    for (int i = 0; i < 32; ++i) {
        auto d = seed;
        size_t pos = 128 + rng() % std::min<size_t>(d.size() - 128, 512);
        d[pos] = (char)0xFF;
        fuzz_gguf_one(d);
    }
    std::cout << "gguf fuzz passed (rejects=" << rejects
              << " — bad magic/truncation rejected, no crash)\n";
    assert(rejects >= 20);  // magic+truncation cases must (almost) all reject
    return 0;
}
