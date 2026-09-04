#include "llm/attention_kv.h"
namespace llm {
void KVCacheAttn::append(const Tensor& k, const Tensor& v){
    k_cache.push_back(k); v_cache.push_back(v); len += k.shape[0];
}
}
