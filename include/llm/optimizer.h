#pragma once
#include "tensor.h"
#include <vector>
namespace llm {
class Optimizer {
public:
    virtual ~Optimizer()=default;
    virtual void step(std::vector<Tensor*>& params, const std::vector<Tensor>& grads)=0;
    virtual void zero_grad()=0;
    virtual void set_lr(float lr) {(void)lr;}
    virtual float get_lr() const { return 0.0f; }
};
class SGD : public Optimizer {
public:
    explicit SGD(float lr=0.01f): lr_(lr){}
    void step(std::vector<Tensor*>& params, const std::vector<Tensor>& grads) override;
    void zero_grad() override {}
    void set_lr(float lr) override { lr_ = lr; }
    float get_lr() const override { return lr_; }
private: float lr_;
};
class Adam : public Optimizer {
public:
    Adam(float lr=1e-3f, float b1=0.9f, float b2=0.999f, float eps=1e-8f);
    void step(std::vector<Tensor*>& params, const std::vector<Tensor>& grads) override;
    void zero_grad() override {}
    void set_lr(float lr) override { lr_ = lr; }
    float get_lr() const override { return lr_; }
private:
    float lr_, b1_, b2_, eps_; int t_=0;
    std::vector<Tensor> m_, v_;
};
class AdamW : public Adam {
public:
    AdamW(float lr=1e-3f, float b1=0.9f, float b2=0.999f, float eps=1e-8f, float wd=0.01f);
    void step(std::vector<Tensor*>& params, const std::vector<Tensor>& grads) override;
private: float wd_;
};
}
