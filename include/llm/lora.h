#pragma once
#include "tensor.h"
namespace llm {
// LoRA adapter: W' = W + A·B * (alpha/r)
// rank r=8 by default, A [in, r], B [r, out] — ~0.1% params
struct LoRAConfig { size_t rank=8; float alpha=16.0f; float dropout=0.0f; };
class LoRAAdapter {
public:
    LoRAAdapter(size_t in_dim, size_t out_dim, LoRAConfig cfg={});
    Tensor forward(const Tensor& x) const; // x [T, in] -> [T, out]
    std::vector<Tensor*> parameters();
private:
    LoRAConfig cfg_;
    Tensor A_, B_; // [in, r], [r, out]
};
}
