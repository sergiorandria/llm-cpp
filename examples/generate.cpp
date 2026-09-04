#include "llm/model.h"
#include "llm/tokenizer.h"
#include "llm/sampling.h"
#include <iostream>
int main(){
    llm::Config cfg; cfg.vocab_size=256; cfg.n_layers=2; cfg.n_heads=4; cfg.n_embd=64; cfg.block_size=128;
    llm::GPT model(cfg);
    llm::Tokenizer tok(256);
    auto ids = tok.encode("Hello");
    auto out = model.generate(ids, 50, 0.8f, 40);
    // apply repetition penalty demo
    std::vector<float> logits(256,0);
    llm::apply_repetition_penalty(logits, out, 1.1f);
    std::cout<<tok.decode(out)<<"\n";
    return 0;
}
