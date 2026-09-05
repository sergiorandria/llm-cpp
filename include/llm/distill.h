#pragma once
#include "model.h"
namespace llm {
// Distillation: teacher 124M -> student 10M, KL divergence on logits, temp=2.0
void distill_step(GPT& student, const GPT& teacher, const std::vector<int>& batch, float temp=2.0f);
}
