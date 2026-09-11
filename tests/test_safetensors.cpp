#include "llm/checkpoint.h"
#include "llm/model.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
    llm::Config cfg;
    cfg.vocab_size = 16; cfg.n_layers = 1; cfg.n_heads = 2; cfg.n_embd = 8; cfg.block_size = 8;
    llm::GPT m1(cfg);
    auto n1 = llm::gpt_named_params(m1);
    llm::save_safetensors("/tmp/st.safetensors", n1);
    llm::GPT m2(cfg);
    // perturb m2 then load
    for (auto p : m2.parameters()) for (auto& v : p->data) v += 1.0f;
    auto n2 = llm::gpt_named_params(m2);
    llm::load_safetensors("/tmp/st.safetensors", n2);
    std::vector<int> toks = {1,2,3,4};
    auto l1 = m1.forward(toks), l2 = m2.forward(toks);
    float md = 0;
    for (size_t i = 0; i < l1.data.size(); ++i) md = std::max(md, std::fabs(l1.data[i]-l2.data[i]));
    std::cout << "st maxd=" << md << "\n";
    assert(md < 1e-6);
    std::cout << "safetensors test passed\n";
    return 0;
}
