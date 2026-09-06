#include "llm/lora.h"
namespace llm {
LoRAAdapter::LoRAAdapter(size_t in_dim, size_t out_dim, LoRAConfig cfg)
    : cfg_(cfg), A_({in_dim, cfg.rank}), B_({cfg.rank, out_dim}) {
    A_.randn(0, 0.02f); B_.fill(0.0f); // B zero-init so initial delta is 0
}
Tensor LoRAAdapter::forward(const Tensor& x) const {
    Tensor h = x.matmul(A_); // [T, r]
    Tensor out = h.matmul(B_); // [T, out]
    float scale = cfg_.alpha / float(cfg_.rank);
    for (auto &v: out.data) v *= scale;
    return out;
}
std::vector<Tensor*> LoRAAdapter::parameters(){ return {&A_, &B_}; }
}
