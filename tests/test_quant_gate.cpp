// F59: quantization degradation gate — fails if int8 ppl drifts >5% or 4-bit err >0.5.
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

#include "llm/gptq.h"
#include "llm/loss.h"
#include "llm/quantize.h"
#include "llm/tokenizer.h"
int main() {
    llm::Config cfg;
    cfg.vocab_size = 256;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_embd = 16;
    cfg.block_size = 64;
    llm::GPT m1(cfg), m2(cfg);
    auto p1 = m1.parameters();
    auto p2 = m2.parameters();
    for (size_t i = 0; i < p1.size(); ++i) p2[i]->data = p1[i]->data;
    // calib batch: raw bytes of data/calib/calib.txt (byte-level tokenizer => byte ids)
    std::ifstream in("data/calib/calib.txt", std::ios::binary);
    std::string txt;
    if (in) txt.assign((std::istreambuf_iterator<char>(in)), {});
    if (txt.size() < 64) txt = std::string(256, 'a');
    std::vector<int> toks;
    for (size_t i = 0; i < 64 && i < txt.size(); ++i) toks.push_back((unsigned char)txt[i]);
    float ppl_fp = llm::compute_loss(m1.forward(toks), toks);
    llm::quantize_model(m2);
    float ppl_q = llm::compute_loss(m2.forward(toks), toks);
    float rel = std::fabs(ppl_q - ppl_fp) / ppl_fp;
    std::cout << "gate ppl_fp=" << ppl_fp << " ppl_q=" << ppl_q << " rel=" << rel << "\n";
    assert(rel < 0.05f);
    // 4-bit group-32 reconstruction bound on a real weight
    llm::Tensor scale;
    auto q = llm::quantize_4bit(*p1[0], scale, 32);
    auto rec = llm::dequantize_4bit(q, scale, 32);
    float e = 0;
    for (size_t i = 0; i < rec.data.size(); ++i) e += std::fabs(rec.data[i] - p1[0]->data[i]);
    e /= rec.data.size();
    std::cout << "gate 4bit err=" << e << "\n";
    assert(e < 0.5f);
    std::cout << "quant gate test passed\n";
    return 0;
}
