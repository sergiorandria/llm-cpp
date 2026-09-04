#include "llm/kv_cache.h"
namespace llm {
KVCache::KVCache(size_t n_layers, size_t max_seq_len, size_t n_embd): n_layers_(n_layers), max_seq_len_(max_seq_len), n_embd_(n_embd){
    k_cache_.assign(n_layers, Tensor({max_seq_len, n_embd},0));
    v_cache_.assign(n_layers, Tensor({max_seq_len, n_embd},0));
}
void KVCache::update(size_t layer, const Tensor& k, const Tensor& v){
    if(layer < k_cache_.size()){
        // naive: append at cur_len
        for(size_t i=0;i<k.shape[0] && cur_len_+i < max_seq_len_; ++i)
            for(size_t j=0;j<n_embd_; ++j) k_cache_[layer](cur_len_+i,j)=k(i,j);
        for(size_t i=0;i<v.shape[0] && cur_len_+i < max_seq_len_; ++i)
            for(size_t j=0;j<n_embd_; ++j) v_cache_[layer](cur_len_+i,j)=v(i,j);
    }
    cur_len_ += k.shape[0];
}
Tensor KVCache::get_k(size_t layer) const { return k_cache_[layer]; }
Tensor KVCache::get_v(size_t layer) const { return v_cache_[layer]; }
void KVCache::clear(){ cur_len_=0; for(auto &t: k_cache_) t.fill(0); for(auto &t: v_cache_) t.fill(0); }
}
