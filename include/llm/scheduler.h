#pragma once
namespace llm {
class LRScheduler {
public:
    virtual ~LRScheduler()=default;
    virtual float get_lr(int step)=0;
};
class CosineScheduler : public LRScheduler {
public:
    CosineScheduler(float base_lr, int warmup, int max_iters);
    float get_lr(int step) override;
private: float base_lr_; int warmup_, max_iters_;
};
class StepScheduler : public LRScheduler {
public:
    StepScheduler(float base_lr, int step_size, float gamma);
    float get_lr(int step) override;
private: float base_lr_; int step_size_; float gamma_;
};
}
