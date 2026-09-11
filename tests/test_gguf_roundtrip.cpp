#include "llm/gguf.h"
#include "llm/model.h"
#include <cassert>
#include <cstdio>
#include <iostream>

int main() {
    llm::Config cfg;
    cfg.vocab_size = 64;
    cfg.n_embd = 16;
    cfg.n_heads = 2;
    cfg.n_layers = 2;
    cfg.block_size = 16;
    cfg.weight_tying = true;

    llm::GPT m1(cfg);
    std::vector<int> prompt = {1, 2, 3, 4};
    auto l1 = m1.forward(prompt);
    // Also test params bit-exact
    auto p1_before = m1.parameters();
    std::vector<llm::FloatVec> snapshot;
    snapshot.reserve(p1_before.size());
    for (auto* p : p1_before) snapshot.push_back(p->data);

    const std::string path = "/tmp/test_gguf_roundtrip.gguf";
    bool ok = llm::save_gguf(m1, path);
    assert(ok && "save_gguf should succeed");

    // Load into new model with same config
    llm::GPT m2(cfg);
    // Ensure m2 initially differs — perturb if deterministic init made them equal
    // wte is tensor_0 [vocab, n_embd]; perturb token 1 which is in prompt
    {
        auto p_mut = m2.parameters();
        if (!p_mut.empty() && p_mut[0]->data.size() > (size_t)cfg.n_embd + 1) {
            size_t idx = 1 * cfg.n_embd; // token 1, dim 0
            p_mut[0]->data[idx] += 1.0f;
        } else if (!p_mut.empty() && !p_mut[0]->data.empty()) {
            p_mut[0]->data[0] += 1.0f;
        }
    }
    auto l2_before = m2.forward(prompt);
    float diff_before = 0;
    for (size_t i = 0; i < l1.data.size(); ++i) diff_before += std::abs(l1.data[i] - l2_before.data[i]);
    std::cout << "pre-load diff " << diff_before << " (must be >1e-3)\n";
    assert(diff_before > 1e-3 && "perturbed init should differ before load");

    bool ok_load = llm::load_gguf(m2, path);
    assert(ok_load && "load_gguf should succeed");

    auto l2 = m2.forward(prompt);
    assert(l1.shape == l2.shape);
    float diff = 0;
    for (size_t i = 0; i < l1.data.size(); ++i) diff += std::abs(l1.data[i] - l2.data[i]);
    std::cout << "gguf roundtrip logits diff " << diff << "\n";
    assert(diff < 1e-5 && "gguf roundtrip should be bit-exact (logits)");

    // Bit-exact param check
    auto p1 = m1.parameters();
    auto p2 = m2.parameters();
    assert(p1.size() == p2.size());
    assert(p1.size() == snapshot.size());
    for (size_t i = 0; i < p1.size(); ++i) {
        assert(p1[i]->shape == p2[i]->shape);
        assert(p1[i]->data.size() == p2[i]->data.size());
        for (size_t j = 0; j < p1[i]->data.size(); ++j) {
            float a = p1[i]->data[j];
            float b = p2[i]->data[j];
            float c = snapshot[i][j];
            assert(std::abs(a - b) < 1e-6 && std::abs(a - c) < 1e-6);
        }
    }
    std::cout << "gguf roundtrip param bit-exact passed (" << p1.size() << " tensors)\n";

    // Mismatch test: load into model with different config should fail
    llm::Config cfg_bad = cfg;
    cfg_bad.n_embd = 32; // different
    llm::GPT m_bad(cfg_bad);
    bool ok_bad = llm::load_gguf(m_bad, path);
    assert(!ok_bad && "load with mismatched config should fail");
    std::cout << "gguf mismatch correctly rejected\n";

    // Bad magic test
    {
        std::FILE* f = std::fopen(path.c_str(), "r+b");
        assert(f);
        std::fseek(f, 0, SEEK_SET);
        uint32_t bad = 0xDEADBEEF;
        std::fwrite(&bad, 4, 1, f);
        std::fclose(f);
        llm::GPT m3(cfg);
        bool ok_bad_magic = llm::load_gguf(m3, path);
        assert(!ok_bad_magic && "bad magic should fail");
        std::cout << "gguf bad magic correctly rejected\n";
    }

    std::remove(path.c_str());
    std::cout << "test_gguf_roundtrip passed\n";
    return 0;
}
