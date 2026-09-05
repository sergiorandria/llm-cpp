#pragma once
#include "model.h"
namespace llm {
struct EvalResult { float mmlu=0, hellaswag=0, ppl=0; };
EvalResult evaluate(const GPT& model, const std::vector<std::vector<int>>& mmlu_data);
}
