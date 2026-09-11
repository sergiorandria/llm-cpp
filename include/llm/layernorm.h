#pragma once
#include "tensor.h"
namespace llm {
class LayerNorm {
public:
    LayerNorm(size_t dim, float eps=1e-5f);
    Tensor forward(const Tensor& x) const;
private:
    Tensor gamma_, beta_;
    float eps_;
};
class RMSNorm {
public:
    RMSNorm(size_t dim, float eps=1e-5f);
    Tensor forward(const Tensor& x) const;
    const Tensor& weight() const { return weight_; }
private:
    Tensor weight_;
    float eps_;
};
}
