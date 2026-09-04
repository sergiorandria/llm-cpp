#include "llm/model.h"
#include <cassert>
#include <iostream>
int main(){
    llm::Config cfg; cfg.n_embd=8; cfg.n_heads=2; cfg.vocab_size=256; cfg.block_size=16; cfg.pos_encoding=llm::PosEncoding::RoPE;
    llm::GPT m(cfg);
    auto out=m.forward({1,2,3});
    assert(out.shape[0]==3);
    std::cout<<"rope test passed\n";
    return 0;
}
