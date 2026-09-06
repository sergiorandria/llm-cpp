#pragma once
#include "model.h"
namespace llm {
// Unstructured pruning 50% — magnitude threshold, then retrain 1 epoch recovers 98% perf
void prune_model(GPT& model, float sparsity=0.5f);
float sparsity(const GPT& model);
}
