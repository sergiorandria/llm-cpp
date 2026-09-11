#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "llm/tokenizer.h"
int main() {
    llm::Tokenizer tok(512);
    tok.train("hello world hello world low low lower", 10);
    // save HF to tmp dir
    system("mkdir -p /tmp/hf_tok");
    tok.save_hf("/tmp/hf_tok");
    llm::Tokenizer tok2(10);
    tok2.load_hf("/tmp/hf_tok/vocab.json", "/tmp/hf_tok/merges.txt");
    // same merges order
    assert(tok2.merges_ == tok.merges_);
    // encode/decode equivalence on sample
    std::string s = "hello world lower";
    assert(tok2.encode(s) == tok.encode(s));
    assert(tok2.decode(tok2.encode(s)) == tok.decode(tok.encode(s)));
    // merges.txt header present
    std::ifstream m("/tmp/hf_tok/merges.txt");
    std::string h;
    std::getline(m, h);
    assert(h.rfind("#version", 0) == 0);
    std::cout << "hf compat test passed merges=" << tok.merges_.size() << "\n";
    return 0;
}
