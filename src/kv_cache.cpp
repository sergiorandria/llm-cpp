#include "llm/kv_cache.h"
namespace llm {
KVCache::KVCache(size_t n_layers, size_t max_seq_len, size_t n_embd, KVCacheConfig cfg): n_layers_(n_layers), max_seq_len_(max_seq_len), n_embd_(n_embd), cfg_(cfg){
    if (cfg_.page_size == 0) cfg_.page_size = 16;
    if(cfg_.paged){
        pages_k_.assign(n_layers, {});
        pages_v_.assign(n_layers, {});
    } else if(cfg_.soa){
        k_cache_.assign(n_layers, Tensor({n_embd, max_seq_len},0));
        v_cache_.assign(n_layers, Tensor({n_embd, max_seq_len},0));
    } else {
        k_cache_.assign(n_layers, Tensor({max_seq_len, n_embd},0));
        v_cache_.assign(n_layers, Tensor({max_seq_len, n_embd},0));
    }
}
void KVCache::ensure_page(size_t layer, size_t page) const {
    auto& pk = const_cast<std::vector<std::vector<Tensor>>&>(pages_k_);
    auto& pv = const_cast<std::vector<std::vector<Tensor>>&>(pages_v_);
    while (pk[layer].size() <= page) {
        pk[layer].emplace_back(Tensor({cfg_.page_size, n_embd_}, 0.0f));
        pv[layer].emplace_back(Tensor({cfg_.page_size, n_embd_}, 0.0f));
    }
}
void KVCache::update(size_t layer, const Tensor& k, const Tensor& v){
    if(layer >= n_layers_) return;
    if(cfg_.paged){
        for(size_t i=0;i<k.shape[0] && cur_len_+i < max_seq_len_; ++i){
            size_t pos = cur_len_+i, pg = pos / cfg_.page_size, off = pos % cfg_.page_size;
            ensure_page(layer, pg);
            for(size_t j=0;j<n_embd_; ++j) pages_k_[layer][pg](off,j)=k(i,j);
        }
        for(size_t i=0;i<v.shape[0] && cur_len_+i < max_seq_len_; ++i){
            size_t pos = cur_len_+i, pg = pos / cfg_.page_size, off = pos % cfg_.page_size;
            ensure_page(layer, pg);
            for(size_t j=0;j<n_embd_; ++j) pages_v_[layer][pg](off,j)=v(i,j);
        }
        return;
    }
    if(layer >= k_cache_.size()) return;
    if(cfg_.soa){
        for(size_t i=0;i<k.shape[0] && cur_len_+i < max_seq_len_; ++i)
            for(size_t j=0;j<n_embd_; ++j) k_cache_[layer](j, cur_len_+i)=k(i,j);
        for(size_t i=0;i<v.shape[0] && cur_len_+i < max_seq_len_; ++i)
            for(size_t j=0;j<n_embd_; ++j) v_cache_[layer](j, cur_len_+i)=v(i,j);
    } else {
        for(size_t i=0;i<k.shape[0] && cur_len_+i < max_seq_len_; ++i)
            for(size_t j=0;j<n_embd_; ++j) k_cache_[layer](cur_len_+i,j)=k(i,j);
        for(size_t i=0;i<v.shape[0] && cur_len_+i < max_seq_len_; ++i)
            for(size_t j=0;j<n_embd_; ++j) v_cache_[layer](cur_len_+i,j)=v(i,j);
    }
}
Tensor KVCache::get_k(size_t layer) const {
    if(cfg_.paged) return get_k_slice(layer);
    return k_cache_[layer];
}
Tensor KVCache::get_v(size_t layer) const {
    if(cfg_.paged) return get_v_slice(layer);
    return v_cache_[layer];
}
Tensor KVCache::get_k_slice(size_t layer) const {
    size_t n = std::min(cur_len_, max_seq_len_);
    if(cfg_.paged){
        Tensor out({n, n_embd_}, 0.0f);
        for(size_t i=0;i<n;++i){
            size_t pg = i / cfg_.page_size, off = i % cfg_.page_size;
            for(size_t j=0;j<n_embd_;++j) out(i,j)=pages_k_[layer][pg](off,j);
        }
        return out;
    }
    if(cfg_.soa){
        Tensor out({n, n_embd_}, 0.0f);
        for(size_t i=0;i<n;++i) for(size_t j=0;j<n_embd_;++j) out(i,j)=k_cache_[layer](j,i);
        return out;
    } else {
        Tensor out({n, n_embd_}, 0.0f);
        for(size_t i=0;i<n;++i) for(size_t j=0;j<n_embd_;++j) out(i,j)=k_cache_[layer](i,j);
        return out;
    }
}
Tensor KVCache::get_v_slice(size_t layer) const {
    size_t n = std::min(cur_len_, max_seq_len_);
    if(cfg_.paged){
        Tensor out({n, n_embd_}, 0.0f);
        for(size_t i=0;i<n;++i){
            size_t pg = i / cfg_.page_size, off = i % cfg_.page_size;
            for(size_t j=0;j<n_embd_;++j) out(i,j)=pages_v_[layer][pg](off,j);
        }
        return out;
    }
    if(cfg_.soa){
        Tensor out({n, n_embd_}, 0.0f);
        for(size_t i=0;i<n;++i) for(size_t j=0;j<n_embd_;++j) out(i,j)=v_cache_[layer](j,i);
        return out;
    } else {
        Tensor out({n, n_embd_}, 0.0f);
        for(size_t i=0;i<n;++i) for(size_t j=0;j<n_embd_;++j) out(i,j)=v_cache_[layer](i,j);
        return out;
    }
}
void KVCache::clear(){ cur_len_=0; for(auto &t: k_cache_) t.fill(0); for(auto &t: v_cache_) t.fill(0); evict(); }
void KVCache::evict(){ for(auto &pl: pages_k_) pl.clear(); for(auto &pl: pages_v_) pl.clear(); }
size_t KVCache::num_pages(size_t layer) const {
    if(!cfg_.paged || layer >= pages_k_.size()) return 0;
    return pages_k_[layer].size();
}
}
