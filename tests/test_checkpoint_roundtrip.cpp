#include "llm/model.h"
#include <cassert>
#include <cstdio>
#include <iostream>

int main() {
    llm::Config cfg; cfg.vocab_size=64; cfg.n_embd=16; cfg.n_heads=2; cfg.n_layers=1; cfg.block_size=16;
    llm::GPT m1(cfg);
    // Save
    const std::string path = "/tmp/test_llm_cpp_roundtrip.bin";
    m1.save(path);
    // Load into new model
    llm::GPT m2(cfg);
    m2.load(path);
    // Verify top-level weights roundtrip (wte, wpe, etc. are serialized in v2)
    // We compare via forward: same input should give same logits after roundtrip
    auto tok = std::vector<int>{1,2,3};
    auto l1 = m1.forward(tok);
    auto l2 = m2.forward(tok);
    assert(l1.shape==l2.shape);
    float diff=0; for(size_t i=0;i<l1.data.size();++i) diff += std::abs(l1.data[i]-l2.data[i]);
    std::cout << "checkpoint roundtrip diff " << diff << "\n";
    // v3 now serializes all tensors incl. TransformerBlocks, so diff must be ~0
    assert(diff < 1e-5 && "v3 checkpoint should be bit-exact (all blocks serialized)");
    assert(m1.num_parameters() == m2.num_parameters());
    // Also verify every parameter tensor is bit-exact
    auto p1 = m1.parameters(); auto p2 = m2.parameters();
    assert(p1.size()==p2.size());
    for(size_t i=0;i<p1.size();++i){
        assert(p1[i]->shape==p2[i]->shape);
        for(size_t j=0;j<p1[i]->data.size();++j) assert(std::abs(p1[i]->data[j]-p2[i]->data[j]) < 1e-6);
    }
    std::cout << "checkpoint roundtrip test passed (v3 all tensors bit-exact)\n";
    std::remove(path.c_str());
    return 0;
}
