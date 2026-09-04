#include "llm/scheduler.h"
#include <cmath>
namespace llm {
CosineScheduler::CosineScheduler(float base_lr, int warmup, int max_iters): base_lr_(base_lr), warmup_(warmup), max_iters_(max_iters){}
float CosineScheduler::get_lr(int step){
    if(step < warmup_) return base_lr_ * step / warmup_;
    float progress = float(step - warmup_) / (max_iters_ - warmup_);
    return base_lr_ * 0.5f * (1 + std::cos(3.14159265f * progress));
}
StepScheduler::StepScheduler(float base_lr, int step_size, float gamma): base_lr_(base_lr), step_size_(step_size), gamma_(gamma){}
float StepScheduler::get_lr(int step){ return base_lr_ * std::pow(gamma_, step / step_size_); }
}
