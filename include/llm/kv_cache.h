#pragma once
#include "tensor.h"
#include <vector>
namespace llm {
class KVCache {
public:
    KVCache(size_t n_layers, size_t max_seq_len, size_t n_embd);
    void update(size_t layer, const Tensor& k, const Tensor& v);
    Tensor get_k(size_t layer) const;
    Tensor get_v(size_t layer) const;
    void clear();
    size_t size() const { return cur_len_; }
private:
    size_t n_layers_, max_seq_len_, n_embd_;
    std::vector<Tensor> k_cache_, v_cache_;
    size_t cur_len_=0;
};
}
