#include "llm/pool.h"
#include "llm/tensor.h"
namespace llm {
size_t TensorPool::key(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (auto s : shape) n *= s;
    return n;
}
Tensor TensorPool::acquire(const std::vector<size_t>& shape) {
    size_t k = key(shape);
    std::lock_guard<std::mutex> lk(mu_);
    auto it = free_.find(k);
    if (it != free_.end() && !it->second.empty()) {
        Tensor t = std::move(it->second.back());
        it->second.pop_back();
        t.reshape(shape);
        t.fill(0.0f);
        hits_++;
        return t;
    }
    misses_++;
    return Tensor(shape, 0.0f);
}
void TensorPool::release(Tensor&& t) {
    if (t.dtype != DType::F32) return;  // only pool F32 activations
    std::lock_guard<std::mutex> lk(mu_);
    free_[t.data.size()].push_back(std::move(t));
}
size_t TensorPool::cached() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = 0;
    for (auto& kv : free_) n += kv.second.size();
    return n;
}
} // namespace llm
