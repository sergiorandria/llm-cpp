#include "llm/kv_cache.h"
#include <cassert>
#include <iostream>
int main(){
    llm::KVCache cache(2, 8, 4);
    cache.clear();
    assert(cache.size()==0);
    llm::Tensor k({1,4}, 1.0f), v({1,4}, 2.0f);
    cache.update(0, k, v);
    cache.update(1, k, v);
    cache.advance(1);
    assert(cache.size()==1);
    auto ks = cache.get_k_slice(0);
    assert(ks.shape[0]==1 && ks(0,0)==1.0f);
    llm::Tensor k2({1,4}, 3.0f), v2({1,4}, 4.0f);
    cache.update(0, k2, v2);
    cache.update(1, k2, v2);
    cache.advance(1);
    assert(cache.size()==2);
    auto ks2 = cache.get_k_slice(0);
    assert(ks2(0,0)==1.0f && ks2(1,0)==3.0f);
    std::cout<<"kv_cache test passed (size "<<cache.size()<<")\n";
    return 0;
}
