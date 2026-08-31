#pragma once
#include "tensor.h"

namespace llm {

class FeedForward {
public:
    FeedForward(size_t n_embd, size_t hidden_dim = 0, bool bias = false);

    Tensor forward(const Tensor& x) const;

private:
    Tensor W1_, W2_;
    Tensor b1_, b2_;
};

} // namespace llm
