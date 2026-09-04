#include "llm/attention.h"
#include <cassert>
#include <iostream>
int main(){
    llm::MultiHeadAttention attn(64,4,32);
    llm::Tensor x({4,64}, 0.5f);
    auto y = attn.forward(x);
    assert(y.shape[0]==4 && y.shape[1]==64);
    std::cout << "attention test passed\n";
    return 0;
}
