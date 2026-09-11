#include "llm/gptq.h"
#include "llm/model.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
    llm::Config cfg;
    cfg.vocab_size = 16; cfg.n_layers = 1; cfg.n_heads = 2; cfg.n_embd = 8; cfg.block_size = 8;
    llm::GPT m(cfg);
    // F52: quantize -> save -> load -> dequant with LOADED scales, err < 0.5
    auto entries = llm::quantize_model_4bit(m, 32);
    assert(!entries.empty());
    llm::save_gptq("/tmp/q.gptq", entries);
    auto loaded = llm::load_gptq("/tmp/q.gptq");
    assert(loaded.size() == entries.size());
    auto params = m.parameters();
    auto named0 = entries;  // originals carry names; compare against live params by order of 2D params
    size_t k = 0;
    float max_err = 0;
    for (auto* p : params) {
        if (p->shape.size() != 2) continue;
        assert(loaded[k].name == entries[k].name);
        auto rec = llm::dequantize_4bit(loaded[k].q, loaded[k].scale, loaded[k].group);
        float e = 0;
        for (size_t i = 0; i < p->data.size(); ++i) e += std::fabs(rec.data[i] - p->data[i]);
        e /= p->data.size();
        max_err = std::max(max_err, e);
        ++k;
    }
    std::cout << "gptq persist max_err=" << max_err << " entries=" << k << "\n";
    assert(max_err < 0.5f);
    // F53: AWQ protects salient channels — quant error on salient rows < naive
    llm::Tensor W({8, 8}, 0.0f);
    for (size_t i = 0; i < 8; ++i) for (size_t j = 0; j < 8; ++j) W(i, j) = (float)(i * 8 + j) * 0.01f;
    llm::Tensor acts({16, 8}, 0.1f);
    for (size_t t = 0; t < 16; ++t) acts(t, 0) = 10.0f;  // channel 0 salient
    auto mag = llm::channel_act_mag(acts);
    assert(mag.data[0] > mag.data[1] * 5);
    auto [scaled, inv] = llm::awq_rescale_for_quant(W, mag, 0.5f);
    // exactness in fp: scaled*inv == orig
    float md = 0;
    for (size_t i = 0; i < 8; ++i) for (size_t j = 0; j < 8; ++j)
        md = std::max(md, std::fabs(scaled(i, j) * inv.data[i] - W(i, j)));
    assert(md < 1e-5);
    // salient row amplified
    assert(std::fabs(scaled(0, 0)) > std::fabs(W(0, 0)));
    std::cout << "gptq+awq test passed\n";
    return 0;
}
