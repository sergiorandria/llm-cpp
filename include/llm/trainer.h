#pragma once
#include "model.h"
#include "dataset.h"
#include "optimizer.h"
#include "scheduler.h"
#include "numpy_utils.h"
#include <string>
namespace llm {
struct TrainConfig {
    size_t batch_size=32;
    size_t max_iters=5000;
    size_t eval_interval=500;
    float grad_clip=1.0f;
    std::string checkpoint_dir="checkpoints";
    size_t grad_accum_steps=1; // gradient accumulation: effective batch = batch_size * accum
    bool checkpointing=false; // activation checkpointing
    bool allow_untrained_params=false; // if false, never silently use random noise for grads
    float label_smoothing=0.0f; // A03: 0 = hard targets
    float loss_scale=1.0f; // A09: >1 for fp16-style scaling (unscaled before optimizer)
    bool deterministic=false; // A08: single-thread + fixed seeds when true
};
bool has_nonfinite(const std::vector<Tensor>& grads); // A07 helper
class Trainer {
public:
    Trainer(GPT& model, const TrainConfig& cfg, Optimizer& optim, LRScheduler& sched);
    void train(Dataset& train_ds, Dataset* val_ds=nullptr);
    float train_step(const std::vector<int>& batch);
    float evaluate(Dataset& ds);
    void save_checkpoint(const std::string& path);
    int skipped_steps() const { return skipped_; }
    int step() const { return step_; }
private:
    GPT& model_;
    TrainConfig cfg_;
    Optimizer& optim_;
    LRScheduler& sched_;
    int step_=0;
    int skipped_=0; // A07: optimizer steps skipped due to non-finite grads
    void clip_grads(std::vector<Tensor>& grads);
};
}
