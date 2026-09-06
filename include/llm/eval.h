#pragma once
#include "model.h"
namespace llm {
// Evaluation harness: perplexity is real (exp(avg cross-entropy)), mmlu/hellaswag are token-accuracy proxies
// computed when data is provided as sequences. For true MMLU/HellaSwag multiple-choice, provide
// data as vector of sequences where accuracy is next-token correctness; bundled samples under data/eval/ can be used.
struct EvalResult { float mmlu=0, hellaswag=0, ppl=0; };
EvalResult evaluate(const GPT& model, const std::vector<std::vector<int>>& data);
}
