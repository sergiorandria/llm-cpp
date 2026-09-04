#include "llm/model.h"
#include <cassert>
#include <iostream>
int main(){
    llm::Config cfg; cfg.vocab_size=256; cfg.n_layers=2; cfg.n_heads=4; cfg.n_embd=64; cfg.block_size=32;
    llm::GPT model(cfg);
    llm::Tokenizer tok(256);
    auto ids = tok.encode("test");
    auto logits = model.forward(ids);
    assert(logits.shape[0]==ids.size());
    assert(logits.shape[1]==256);
    std::cout << "model test passed params=" << model.num_parameters() << "\n";
    return 0;
}
