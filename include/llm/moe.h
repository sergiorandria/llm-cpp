#pragma once
#include "tensor.h"
namespace llm {
// MoE FFN: 8 experts, top-2 routing, load balancing loss
// Sparsity 75% — only 2/8 experts active per token
struct MoEConfig {
    size_t n_experts = 8;
    size_t top_k = 2;
    float capacity_factor = 1.25f;
};
class MoEFFN {
   public:
    MoEFFN(size_t n_embd, MoEConfig cfg = {});
    Tensor forward(const Tensor& x) const;
    // C26: load-balancing aux loss — CV^2 of gate probs (0 = uniform, >0 = collapsed)
    float aux_loss(const Tensor& x, float coef = 0.01f) const;

   private:
    MoEConfig cfg_;
    std::vector<Tensor> experts_W1_, experts_W2_;
    Tensor gate_;  // [n_embd, n_experts]
};
}  // namespace llm
