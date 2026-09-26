#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

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
    // Q4_K super-block codec roundtrips (deterministic pseudo-random data).
    // Explicit checks (not assert-only): Release NDEBUG elides assert bodies,
    // and these bounds must hold in every build type.
    {
        auto fail = [](const char* msg) {
            std::cerr << "FAIL: " << msg << "\n";
            return 1;
        };
        auto lcg = [](unsigned& s) {
            s = s * 1664525u + 1013904223u;
            return (float)(s >> 8) / (float)(1u << 24);
        };
        // randn-ish centered weights (typical case: exact-ish sub-mins)
        {
            unsigned s = 42;
            float x[256], y[256];
            for (int i = 0; i < 256; ++i) x[i] = (lcg(s) + lcg(s) + lcg(s) - 1.5f) * 0.4f;
            uint8_t blk[144];
            llm::encode_q4k_block(x, blk);
            llm::decode_q4k_block(blk, y);
            float md = 0;
            for (int i = 0; i < 256; ++i) md = std::max(md, std::fabs(x[i] - y[i]));
            std::cout << "q4k centered maxd=" << md << "\n";
            if (!(md < 0.15f)) return fail("q4k centered bound");
        }
        // all-positive offset block (min-anchor sign path)
        {
            float x[256], y[256];
            for (int i = 0; i < 256; ++i) x[i] = 100.0f + (float)(i % 37) * 0.5f;
            uint8_t blk[144];
            llm::encode_q4k_block(x, blk);
            llm::decode_q4k_block(blk, y);
            float md = 0;
            for (int i = 0; i < 256; ++i) md = std::max(md, std::fabs(x[i] - y[i]));
            std::cout << "q4k offset maxd=" << md << "\n";
            if (!(md < 1.0f)) return fail("q4k offset bound");
        }
        // constant +1 (norm-weight case) must survive exactly-ish
        {
            float x[256], y[256];
            for (int i = 0; i < 256; ++i) x[i] = 1.0f;
            uint8_t blk[144];
            llm::encode_q4k_block(x, blk);
            llm::decode_q4k_block(blk, y);
            float md = 0;
            for (int i = 0; i < 256; ++i) md = std::max(md, std::fabs(x[i] - y[i]));
            std::cout << "q4k const maxd=" << md << "\n";
            if (!(md < 0.05f)) return fail("q4k const bound");
        }
        // zeros exact; non-finite block -> zeros, never NaN
        {
            float x[256] = {0}, y[256];
            uint8_t blk[144];
            llm::encode_q4k_block(x, blk);
            llm::decode_q4k_block(blk, y);
            for (int i = 0; i < 256; ++i)
                if (y[i] != 0.0f) return fail("q4k zeros not exact");
            x[7] = std::numeric_limits<float>::quiet_NaN();
            llm::encode_q4k_block(x, blk);
            llm::decode_q4k_block(blk, y);
            for (int i = 0; i < 256; ++i)
                if (y[i] != 0.0f) return fail("q4k NaN block not zeroed");
        }
    }
    llm::Config cfg;
    cfg.vocab_size = 32;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_embd = 32;
    cfg.block_size = 16;
    // Q8_0 roundtrip err < 0.05 (dequant error), greedy decode identical
    // NOTE: calls hoisted out of assert() — NDEBUG elides assert args in Release.
    // Q4_K (qtype 12): super-block hierarchy beats Q4_0 flat blocks; same gate.
    for (int qt : {8, 4, 12}) {
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
        // Q4_K bound must hold in every build type (Release elides asserts).
        if (qt == 12 && !(md < 0.5f)) {
            std::cerr << "FAIL: q4k model roundtrip bound\n";
            return 1;
        }
        std::vector<int> prompt = {1, 2, 3};
        auto g1 = m1.generate(prompt, 6, 0.0f);
        auto g2 = m2.generate(prompt, 6, 0.0f);
        assert(g1 == g2);
    }
    std::cout << "gguf quant test passed\n";
    return 0;
}
