#include "llm/quantize.h"
#include <algorithm>
namespace llm {
Tensor quantize_int8(const Tensor& x){
    float maxv=0; for(float v: x.data) maxv=std::max(maxv, std::abs(v));
    float scale = maxv / 127.0f;
    Tensor q(x.shape,0);
    for(size_t i=0;i<x.data.size();++i) q.data[i]= std::round(x.data[i]/ (scale+1e-8f));
    return q;
}
Tensor dequantize_int8(const Tensor& q, float scale){
    Tensor out(q.shape,0);
    for(size_t i=0;i<q.data.size();++i) out.data[i]=q.data[i]*scale;
    return out;
}
}
