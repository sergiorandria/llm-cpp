#pragma once
#include "tensor.h"

namespace llm {

enum class Activation { GELU, SILU, RELU };

class FeedForward {
public:
    FeedForward(size_t n_embd, size_t hidden_dim = 0, bool bias = false, Activation act = Activation::GELU);

    Tensor forward(const Tensor& x) const;
    Tensor backward(const Tensor& x, const Tensor& grad_out) const;
    std::vector<Tensor*> parameters();
    std::vector<const Tensor*> parameters() const;

private:
    Tensor W1_, W2_;
    Tensor W3_; // for SwiGLU gated
    Activation act_;
    Tensor b1_, b2_;
};

} // namespace llm
