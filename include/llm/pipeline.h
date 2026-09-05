#pragma once
#include "model.h"
namespace llm {
// Pipeline parallel: split n_layers across stages, micro-batch 4
// For 12 layers → 3 stages ×4 layers, overlap comm with compute
class Pipeline {
public:
    Pipeline(GPT& model, size_t stages=3, size_t micro_batch=4);
    void train_step(const std::vector<int>& batch);
private:
    GPT& model_;
    size_t stages_, micro_;
};
}
