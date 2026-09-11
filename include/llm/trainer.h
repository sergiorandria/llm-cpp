#pragma once
#include <string>

#include "dataset.h"
#include "model.h"
#include "numpy_utils.h"
#include "optimizer.h"
#include "scheduler.h"
namespace llm {
struct TrainConfig {
    size_t batch_size = 32;
    size_t max_iters = 5000;
    size_t eval_interval = 500;
    float grad_clip = 1.0f;
    std::string checkpoint_dir = "checkpoints";
    size_t grad_accum_steps = 1;  // gradient accumulation: effective batch = batch_size * accum
    bool checkpointing = false;   // activation checkpointing
    bool allow_untrained_params = false;  // if false, never silently use random noise for grads
    float label_smoothing = 0.0f;         // A03: 0 = hard targets
    float loss_scale = 1.0f;     // A09: >1 for fp16-style scaling (unscaled before optimizer)
    bool deterministic = false;  // A08: single-thread + fixed seeds when true
};
bool has_nonfinite(const std::vector<Tensor>& grads);  // A07 helper
// D36: average per-shard grads (single-node allreduce mean). All entries must share shapes.
void mean_reduce_grads(const std::vector<std::vector<Tensor>>& shard_grads,
                       std::vector<Tensor>& out);
class Trainer {
   public:
    Trainer(GPT& model, const TrainConfig& cfg, Optimizer& optim, LRScheduler& sched);
    void train(Dataset& train_ds, Dataset* val_ds = nullptr);
    float train_step(const std::vector<int>& batch);
    // D35: accumulate grads over grad_accum_steps micro-batches before stepping.
    // Returns mean loss; steps optimizer only every K calls (K=cfg.grad_accum_steps).
    float train_step_accum(const std::vector<int>& micro_batch);
    float evaluate(Dataset& ds);
    void save_checkpoint(const std::string& path);
    // D31: full training-state save/load (model weights + optimizer + step/rng).
    // Writes <prefix>.model.bin, <prefix>.opt.bin, <prefix>.json
    void save_train_state(const std::string& prefix);
    void load_train_state(const std::string& prefix);
    int skipped_steps() const {
        return skipped_;
    }
    int step() const {
        return step_;
    }
    void set_step(int s) {
        step_ = s;
    }
    // H79: NaN rollback — snapshot last-good params; on non-finite loss restore + halve LR
    void snapshot_params();
    bool rollback_if_nonfinite(float loss);  // true if rolled back
    // D37: last per-step metrics (JSONL logger reads these)
    float last_loss() const {
        return last_loss_;
    }
    float last_grad_norm() const {
        return last_grad_norm_;
    }

   private:
    GPT& model_;
    TrainConfig cfg_;
    Optimizer& optim_;
    LRScheduler& sched_;
    int step_ = 0;
    int skipped_ = 0;  // A07: optimizer steps skipped due to non-finite grads
    // D35 accumulation buffer (sum of micro-batch grads) + counter
    std::vector<Tensor> accum_;
    size_t accum_count_ = 0;
    float last_loss_ = 0.0f;
    float last_grad_norm_ = 0.0f;
    std::vector<Tensor> last_good_;  // H79 snapshot
    bool has_snapshot_ = false;
    void clip_grads(std::vector<Tensor>& grads);
};
}  // namespace llm
