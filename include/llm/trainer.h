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
};
class Trainer {
public:
    Trainer(GPT& model, const TrainConfig& cfg, Optimizer& optim, LRScheduler& sched);
    void train(Dataset& train_ds, Dataset* val_ds=nullptr);
    float train_step(const std::vector<int>& batch);
    float evaluate(Dataset& ds);
    void save_checkpoint(const std::string& path);
private:
    GPT& model_;
    TrainConfig cfg_;
    Optimizer& optim_;
    LRScheduler& sched_;
    int step_=0;
    void clip_grads(std::vector<Tensor>& grads);
};
}
