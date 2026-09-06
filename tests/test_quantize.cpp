#include "llm/quantize.h"
#include <cassert>
#include <iostream>
int main(){
    llm::Tensor x({4,4}, 0.5f);
    for(size_t i=0;i<x.data.size();++i) x.data[i]= (i%2?0.5f:-0.5f)*(i*0.1f);
    auto qt = llm::quantize_with_scale(x);
    float err = llm::quantize_error(x, qt);
    std::cout<<"quant scale "<<qt.scale<<" err "<<err<<"\n";
    assert(err < 0.1f);
    auto rec = llm::dequantize(qt);
    assert(rec.shape==x.shape);
    std::cout<<"quant test passed\n";
    return 0;
}
