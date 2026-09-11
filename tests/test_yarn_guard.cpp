#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/config.h"
#include "llm/model.h"
int main() {
    // C23: NTK vs YaRN differ but both finite at 8k-style scaling
    llm::Config base;
    base.vocab_size = 16;
    base.n_layers = 1;
    base.n_heads = 2;
    base.n_embd = 8;
    base.block_size = 32;
    base.pos_encoding = llm::PosEncoding::RoPE;
    base.rope_theta = 50000.0f;
    base.rope_scaling = 8.0f;
    base.rope_mode = 0;
    llm::GPT m_ntk(base);
    base.rope_mode = 1;
    llm::GPT m_yarn(base);
    // copy weights for fair comparison
    auto pn = m_ntk.parameters();
    auto py = m_yarn.parameters();
    for (size_t i = 0; i < pn.size(); ++i) py[i]->data = pn[i]->data;
    std::vector<int> toks(16);
    for (int i = 0; i < 16; ++i) toks[i] = i;
    auto ln = m_ntk.forward(toks), ly = m_yarn.forward(toks);
    for (auto v : ln.data) assert(std::isfinite(v));
    for (auto v : ly.data) assert(std::isfinite(v));
    float md = 0;
    for (size_t i = 0; i < ln.data.size(); ++i)
        md = std::max(md, std::fabs(ln.data[i] - ly.data[i]));
    std::cout << "ntk-vs-yarn maxd=" << md << "\n";
    assert(md > 1e-6);  // modes genuinely differ
    // C24: guard
    llm::Config bad = base;
    bad.use_alibi = true;
    bad.pos_encoding = llm::PosEncoding::RoPE;
    assert(!llm::validate_config(bad));
    bad.pos_encoding = llm::PosEncoding::Learned;
    assert(llm::validate_config(bad));
    std::cout << "yarn+guard test passed\n";
    return 0;
}
