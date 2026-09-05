#include "llm/kv_cache.h"
namespace llm {
KVCache::KVCache(size_t n_layers, size_t max_seq_len, size_t n_embd): n_layers_(n_layers), max_seq_len_(max_seq_len), n_embd_(n_embd){
    k_cache_.assign(n_layers, Tensor({max_seq_len, n_embd},0));
    v_cache_.assign(n_layers, Tensor({max_seq_len, n_embd},0));
}
void KVCache::update(size_t layer, const Tensor& k, const Tensor& v){
    if(layer >= k_cache_.size()) return;
    for(size_t i=0;i<k.shape[0] && cur_len_+i < max_seq_len_; ++i)
        for(size_t j=0;j<n_embd_; ++j) k_cache_[layer](cur_len_+i,j)=k(i,j);
    for(size_t i=0;i<v.shape[0] && cur_len_+i < max_seq_len_; ++i)
        for(size_t j=0;j<n_embd_; ++j) v_cache_[layer](cur_len_+i,j)=v(i,j);
    // Don't auto-increment cur_len_ here — caller advances once per token after all layers
}
Tensor KVCache::get_k(size_t layer) const { return k_cache_[layer]; }
Tensor KVCache::get_v(size_t layer) const { return v_cache_[layer]; }
Tensor KVCache::get_k_slice(size_t layer) const {
    size_t n = std::min(cur_len_, max_seq_len_);
    Tensor out({n, n_embd_}, 0.0f);
    for(size_t i=0;i<n;++i) for(size_t j=0;j<n_embd_;++j) out(i,j)=k_cache_[layer](i,j);
    return out;
}
Tensor KVCache::get_v_slice(size_t layer) const {
    size_t n = std::min(cur_len_, max_seq_len_);
    Tensor out({n, n_embd_}, 0.0f);
    for(size_t i=0;i<n;++i) for(size_t j=0;j<n_embd_;++j) out(i,j)=v_cache_[layer](i,j);
    return out;
}
void KVCache::clear(){ cur_len_=0; for(auto &t: k_cache_) t.fill(0); for(auto &t: v_cache_) t.fill(0); }
}
