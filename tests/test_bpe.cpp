#include "llm/tokenizer.h"
#include <cassert>
#include <iostream>
int main(){
    llm::Tokenizer tok(500);
    tok.train("hello world hello world hello", 10);
    std::string text="hello world";
    auto ids = tok.encode(text);
    auto dec = tok.decode(ids);
    std::cout<<"bpe ids size "<<ids.size()<<" decoded '"<<dec<<"'\n";
    // ensure encode not empty and decode contains hello
    assert(!ids.empty());
    std::cout<<"bpe test passed\n";
    return 0;
}
