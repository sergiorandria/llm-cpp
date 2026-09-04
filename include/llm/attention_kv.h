#pragma once
#include "tensor.h"
#include <vector>
namespace llm {
struct KVCacheAttn {
    std::vector<Tensor> k_cache, v_cache;
    size_t len=0;
    void append(const Tensor& k, const Tensor& v);
    void clear(){ k_cache.clear(); v_cache.clear(); len=0; }
};
}
