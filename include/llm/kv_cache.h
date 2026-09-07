#pragma once
#include "tensor.h"
#include <vector>
namespace llm {
struct KVCacheConfig { bool soa = false; }; // SoA [n_embd, max_seq_len] better for head slices
class KVCache {
public:
    KVCache(size_t n_layers, size_t max_seq_len, size_t n_embd, KVCacheConfig cfg = {});
    void update(size_t layer, const Tensor& k, const Tensor& v);
    Tensor get_k(size_t layer) const;
    Tensor get_v(size_t layer) const;
    // Return K/V sliced to current seq len (not full max_seq_len)
    Tensor get_k_slice(size_t layer) const;
    Tensor get_v_slice(size_t layer) const;
    void clear();
    size_t size() const { return cur_len_; }
    void set_size(size_t n) { cur_len_ = std::min(n, max_seq_len_); }
    void advance(size_t n) { cur_len_ = std::min(cur_len_ + n, max_seq_len_); }
    bool is_soa() const { return cfg_.soa; }
private:
    size_t n_layers_, max_seq_len_, n_embd_;
    KVCacheConfig cfg_;
    std::vector<Tensor> k_cache_, v_cache_;
    size_t cur_len_=0;
};
}
