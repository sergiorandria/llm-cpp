#include "llm/transformer.h"
#include <cassert>
#include <iostream>
int main(){
    llm::TransformerBlock blk(64,4,32);
    llm::Tensor x({8,64}, 0.1f);
    auto y = blk.forward(x);
    assert(y.shape[0]==8 && y.shape[1]==64);
    std::cout << "transformer test passed\n";
    return 0;
}
