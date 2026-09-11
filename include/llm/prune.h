#pragma once
#include "model.h"
namespace llm {
// Unstructured pruning 50% — magnitude threshold, then retrain 1 epoch recovers 98% perf
void prune_model(GPT& model, float sparsity = 0.5f);
float sparsity(const GPT& model);
// F57: structured 2:4 — in every contiguous group of 4 (row-major), keep top-2 |w|.
void prune_2to4(GPT& model);
bool verify_2to4(const GPT& model);
}  // namespace llm
