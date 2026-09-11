#include "llm/prune.h"
#include "llm/loss.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
    // F55: sparse kernel == dense on pruned weight
    llm::Tensor A({4, 8}, 0.0f), W({8, 6}, 0.0f);
    for (size_t i = 0; i < A.data.size(); ++i) A.data[i] = (float)(i % 5) * 0.3f - 0.4f;
    for (size_t i = 0; i < W.data.size(); ++i) W.data[i] = (i % 2) ? 0.0f : (float)i * 0.05f;  // 50% zeros
    auto y_sp = A.matmul_sparse(W);
    auto y_dn = A.matmul(W);
    float md = 0;
    for (size_t i = 0; i < y_sp.data.size(); ++i) md = std::max(md, std::fabs(y_sp.data[i]-y_dn.data[i]));
    std::cout << "sparse maxd=" << md << "\n";
    assert(md < 1e-4f);
    // F57: 2:4 property + ppl measured
    llm::Config cfg;
    cfg.vocab_size = 16; cfg.n_layers = 1; cfg.n_heads = 2; cfg.n_embd = 8; cfg.block_size = 8;
    llm::GPT m(cfg);
    std::vector<int> toks = {1,2,3,4,5,6,7,8};
    float ppl_before = llm::compute_loss(m.forward(toks), toks);
    llm::prune_2to4(m);
    bool ok = llm::verify_2to4(m);
    assert(ok);
    float ppl_after = llm::compute_loss(m.forward(toks), toks);
    std::cout << "ppl " << ppl_before << " -> " << ppl_after << "\n";
    assert(std::isfinite(ppl_after));
    std::cout << "sparse+24 test passed\n";
    return 0;
}
