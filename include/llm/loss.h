#pragma once
#include "tensor.h"
namespace llm {
float cross_entropy_loss(const Tensor& logits, const std::vector<int>& targets);
float compute_loss(const Tensor& logits, const std::vector<int>& targets);
}
