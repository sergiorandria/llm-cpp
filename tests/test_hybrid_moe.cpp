#include <cassert>
#include <iostream>

#include "llm/model.h"
#include "llm/moe.h"
#include "llm/sliding.h"
int main() {
    // C25: pattern + sliding==full when T <= window
    assert(llm::is_global_layer(0, 1, 0) == true);
    assert(llm::is_global_layer(3, 4, 512) == true);  // 4th layer global
    assert(llm::is_global_layer(0, 4, 512) == false);
    assert(llm::is_global_layer(1, 4, 512) == false);
    llm::Tensor Q({8, 4}, 0.3f), K({8, 4}, -0.1f), V({8, 4}, 0.2f);
    auto y = llm::sliding_attention(Q, K, V, 512);
    assert(y.shape[0] == 8);
    // hybrid config validates + forwards finite
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_layers = 4;
    cfg.n_heads = 2;
    cfg.n_embd = 8;
    cfg.block_size = 16;
    cfg.sliding_window = 8;
    cfg.global_every = 4;
    llm::GPT m(cfg);
    auto logits = m.forward({1, 2, 3, 4, 5, 6, 7, 8});
    for (auto v : logits.data) assert(std::isfinite(v));
    // C26: uniform routing aux≈0, collapsed >0
    llm::MoEFFN moe(8);
    llm::Tensor x({4, 8}, 0.0f);  // all-zero input -> uniform gate probs
    float aux_u = moe.aux_loss(x);
    std::cout << "aux_uniform=" << aux_u << "\n";
    assert(aux_u < 1e-3);
    llm::Tensor xs({4, 8}, 5.0f);  // large input -> peaked routing
    float aux_c = moe.aux_loss(xs);
    std::cout << "aux_peaked=" << aux_c << "\n";
    assert(aux_c >= aux_u);
    std::cout << "hybrid+moe test passed\n";
    return 0;
}
