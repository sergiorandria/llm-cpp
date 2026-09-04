#include "llm/embedding.h"
#include <cassert>
#include <iostream>
int main(){
    llm::Embedding emb(256,32);
    auto out = emb.forward({1,2,3});
    assert(out.shape[0]==3 && out.shape[1]==32);
    std::cout<<"embedding test passed\n";
    return 0;
}
