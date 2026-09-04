#include "llm/tensor.h"
#include <cassert>
#include <iostream>
int main(){
    llm::Tensor a({2,3}, 1.0f);
    llm::Tensor b({3,2}, 2.0f);
    auto c = a.matmul(b);
    assert(c.shape[0]==2 && c.shape[1]==2);
    assert(c(0,0)==6.0f);
    auto sm = c.softmax(1);
    auto ln = c.layernorm();
    std::cout << "tensor tests passed\n";
    return 0;
}
