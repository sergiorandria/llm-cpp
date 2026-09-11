#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/lora.h"
#include "llm/model.h"
int main() {
    // C27: merge/unmerge roundtrip + merged forward == base+adapter
    llm::LoRAAdapter ad(8, 8, llm::LoRAConfig{4, 8.0f});
    llm::Tensor W({8, 8}, 0.5f);
    llm::Tensor W0 = W;
    llm::Tensor x({2, 8}, 0.3f);
    // base+adapter path
    llm::Tensor y_base = x.matmul(W0);
    llm::Tensor y_ad = ad.forward(x);
    llm::Tensor y_sum = y_base;
    for (size_t i = 0; i < y_sum.data.size(); ++i) y_sum.data[i] += y_ad.data[i];
    ad.merge_into(W);
    llm::Tensor y_merged = x.matmul(W);
    float md = 0;
    for (size_t i = 0; i < y_sum.data.size(); ++i)
        md = std::max(md, std::fabs(y_sum.data[i] - y_merged.data[i]));
    std::cout << "lora merge maxd=" << md << "\n";
    assert(md < 1e-4);
    ad.unmerge_from(W);
    for (size_t i = 0; i < W.data.size(); ++i) assert(std::fabs(W.data[i] - W0.data[i]) < 1e-6);
    // only adapter params receive grad: adapter has 2 tensors, W untouched by adapter API
    assert(ad.parameters().size() == 2);
    // C28: copy-on-tie semantics locked — tie copies wte^T into lm_head, params counted once
    llm::Config cfg;
    cfg.vocab_size = 8;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_embd = 4;
    cfg.block_size = 8;
    cfg.weight_tying = true;
    llm::GPT m(cfg);
    m.tie_weights();
    auto logits = m.forward({1, 2, 3});
    for (auto v : logits.data) assert(std::isfinite(v));
    size_t n_tied = m.num_parameters();
    cfg.weight_tying = false;
    llm::GPT m2(cfg);
    size_t n_untied = m2.num_parameters();
    assert(n_untied > n_tied);  // tied excludes duplicate lm_head
    std::cout << "lora+tying test passed tied=" << n_tied << " untied=" << n_untied << "\n";
    return 0;
}
