#pragma once
#include "tensor.h"
namespace llm {
float cross_entropy_loss(const Tensor& logits, const std::vector<int>& targets);
float compute_loss(const Tensor& logits, const std::vector<int>& targets);
// dlogits for softmax-CE, mean over T. smoothing in [0,1): 0 = hard targets.
Tensor cross_entropy_backward(const Tensor& logits, const std::vector<int>& targets,
                              float smoothing = 0.0f);
}  // namespace llm
