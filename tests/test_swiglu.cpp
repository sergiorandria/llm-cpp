#include "llm/transformer.h"
#include <cassert>
#include <iostream>
int main(){
    llm::FeedForward ffn(32, 0, false, llm::Activation::SILU);
    llm::Tensor x({4,32}, 0.5f);
    auto y=ffn.forward(x);
    assert(y.shape[0]==4 && y.shape[1]==32);
    std::cout<<"swiglu test passed\n";
    return 0;
}
