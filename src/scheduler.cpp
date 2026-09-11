#include "llm/scheduler.h"

#include <cmath>
namespace llm {
CosineScheduler::CosineScheduler(float base_lr, int warmup, int max_iters)
    : base_lr_(base_lr), warmup_(warmup), max_iters_(max_iters) {}
CosineScheduler::CosineScheduler(float base_lr, int warmup, int max_iters, float min_lr, int cycles)
    : base_lr_(base_lr),
      warmup_(warmup),
      max_iters_(max_iters),
      min_lr_(min_lr),
      cycles_(cycles > 0 ? cycles : 1) {}
float CosineScheduler::get_lr(int step) {
    if (step < warmup_) return base_lr_ * step / (warmup_ > 0 ? warmup_ : 1);
    int span = max_iters_ - warmup_;
    if (span <= 0) return min_lr_;
    if (cycles_ > 1) {
        // restarts: map step into current cycle
        int cyc_len = span / cycles_;
        if (cyc_len <= 0) cyc_len = span;
        int local = (step - warmup_) % cyc_len;
        float progress = float(local) / cyc_len;
        return min_lr_ + (base_lr_ - min_lr_) * 0.5f * (1 + std::cos(3.14159265f * progress));
    }
    float progress = float(step - warmup_) / span;
    if (progress > 1) progress = 1;
    return min_lr_ + (base_lr_ - min_lr_) * 0.5f * (1 + std::cos(3.14159265f * progress));
}
StepScheduler::StepScheduler(float base_lr, int step_size, float gamma)
    : base_lr_(base_lr), step_size_(step_size), gamma_(gamma) {}
float StepScheduler::get_lr(int step) {
    return base_lr_ * std::pow(gamma_, step / step_size_);
}
}  // namespace llm
