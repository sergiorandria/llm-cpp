#include "llm/tokenizer.h"
#include <cassert>
#include <iostream>
int main(){
    llm::Tokenizer tok(256);
    auto ids = tok.encode("hello");
    auto s = tok.decode(ids);
    assert(s=="hello");
    std::cout << "tokenizer test passed\n";
    return 0;
}
