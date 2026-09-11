#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/gguf.h"
int main() {
    // block codec roundtrips
    {
        float x[32], y[32];
        for (int i = 0; i < 32; ++i) x[i] = (float)i * 0.1f - 1.5f;
        float sc;
        int8_t q[32];
        llm::encode_q80_block(x, sc, q);
        llm::decode_q80_block(sc, q, y);
        float md = 0;
        for (int i = 0; i < 32; ++i) md = std::max(md, std::fabs(x[i] - y[i]));
        std::cout << "q80 block maxd=" << md << "\n";
        assert(md < 0.02f);
        float s4;
        uint8_t p[16];
        float z[32];
        llm::encode_q40_block(x, s4, p);
        llm::decode_q40_block(s4, p, z);
        md = 0;
        for (int i = 0; i < 32; ++i) md = std::max(md, std::fabs(x[i] - z[i]));
        std::cout << "q40 block maxd=" << md << "\n";
        assert(md < 0.2f);
    }
    llm::Config cfg;
    cfg.vocab_size = 32;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_embd = 32;
    cfg.block_size = 16;
    // Q8_0 roundtrip err < 0.05 (dequant error), greedy decode identical
    // NOTE: calls hoisted out of assert() — NDEBUG elides assert args in Release.
    for (int qt : {8, 4}) {
        llm::GPT m1(cfg), m2(cfg);
        for (auto p : m2.parameters())
            for (auto& v : p->data) v += 1.0f;  // poison m2
        bool saved = llm::save_gguf_quant(m1, "/tmp/q.gguf", qt);
        bool loaded = llm::load_gguf(m2, "/tmp/q.gguf");
        assert(saved && loaded);
        float md = 0;
        auto p1 = m1.parameters();
        auto p2 = m2.parameters();
        for (size_t i = 0; i < p1.size(); ++i)
            for (size_t j = 0; j < p1[i]->data.size(); ++j)
                md = std::max(md, std::fabs(p1[i]->data[j] - p2[i]->data[j]));
        std::cout << "q" << qt << " param maxd=" << md << "\n";
        assert(md < 0.5f);
        if (qt == 8) assert(md < 0.05f);
        std::vector<int> prompt = {1, 2, 3};
        auto g1 = m1.generate(prompt, 6, 0.0f);
        auto g2 = m2.generate(prompt, 6, 0.0f);
        assert(g1 == g2);
    }
    std::cout << "gguf quant test passed\n";
    return 0;
}
