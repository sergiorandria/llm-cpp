#include "llm/quantize.h"
#include "llm/loss.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
    // kernel equivalence: int8 fused == dequant-then-matmul (up to fp assoc error)
    llm::Tensor A({4, 8}, 0.0f), W({8, 6}, 0.0f);
    for (size_t i = 0; i < A.data.size(); ++i) A.data[i] = (float)(i % 7) * 0.2f - 0.5f;
    for (size_t i = 0; i < W.data.size(); ++i) W.data[i] = (float)(i % 9) * 0.1f - 0.3f;
    auto wq = llm::quantize_with_scale(W);
    auto y_fused = llm::matmul_int8(A, wq);
    auto y_ref = A.matmul(llm::dequantize(wq));
    float md = 0;
    for (size_t i = 0; i < y_fused.data.size(); ++i) md = std::max(md, std::fabs(y_fused.data[i]-y_ref.data[i]));
    std::cout << "int8 kernel maxd=" << md << "\n";
    assert(md < 1e-4);
    // model-level: int8 ppl within 2% of fp32 on tiny model
    llm::Config cfg;
    cfg.vocab_size = 16; cfg.n_layers = 1; cfg.n_heads = 2; cfg.n_embd = 8; cfg.block_size = 8;
    llm::GPT m1(cfg), m2(cfg);
    auto p1 = m1.parameters(); auto p2 = m2.parameters();
    for (size_t i = 0; i < p1.size(); ++i) p2[i]->data = p1[i]->data;
    std::vector<int> toks = {1,2,3,4,5,6,7,8};
    float ppl_fp = llm::compute_loss(m1.forward(toks), toks);
    llm::quantize_model(m2);
    float ppl_q = llm::compute_loss(m2.forward(toks), toks);
    std::cout << "ppl_fp=" << ppl_fp << " ppl_q=" << ppl_q << "\n";
    assert(std::fabs(ppl_q - ppl_fp) / ppl_fp < 0.02f);
    std::cout << "int8 infer test passed\n";
    return 0;
}
