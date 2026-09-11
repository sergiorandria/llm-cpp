#pragma once
#include "model.h"
namespace llm {
// Distillation: teacher 124M -> student 10M, KL divergence on logits, temp=2.0
void distill_step(GPT& student, const GPT& teacher, const std::vector<int>& batch, float temp=2.0f);
// F56: cache teacher logits to disk once (binary: n_batches, then per batch T,V + floats).
// cache_teacher_logits runs teacher.forward per batch and appends; load_cached_logits
// reads all batches back. Cached KL grads match live within 1e-6.
void cache_teacher_logits(const GPT& teacher, const std::vector<std::vector<int>>& batches,
                          const std::string& path);
std::vector<Tensor> load_cached_logits(const std::string& path);
void distill_step_cached(GPT& student, const Tensor& teacher_logits, const std::vector<int>& batch,
                         float temp=2.0f);
}
