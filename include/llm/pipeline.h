#pragma once
#include "model.h"
namespace llm {
// Pipeline (gradient accumulation with staged execution): split n_layers across stages and micro-batches
// For 12 layers → 3 stages ×4 layers. Single-process staged execution (no true multi-device parallelism yet),
// but verifies stage partitioning and uses shared optimizer. For true multi-device, overlap comm with compute would be added.
class Pipeline {
public:
    Pipeline(GPT& model, size_t stages=3, size_t micro_batch=4);
    void train_step(const std::vector<int>& batch);
    // Honest alias for single-process: this is gradient accumulation, not true pipeline parallelism
    using GradientAccumulator = Pipeline;
private:
    GPT& model_;
    size_t stages_, micro_;
};
}
