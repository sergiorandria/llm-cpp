#include "llm/layernorm.h"
#include <cassert>
#include <iostream>
int main(){
    llm::LayerNorm ln(8);
    llm::Tensor x({2,8}, 1.0f);
    auto y = ln.forward(x);
    assert(y.shape[0]==2);
    std::cout<<"layernorm test passed\n";
    return 0;
}
