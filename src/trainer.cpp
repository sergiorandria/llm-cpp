#include "llm/trainer.h"
#include "llm/loss.h"
#include <iostream>
#include <cmath>
namespace llm {
Trainer::Trainer(GPT& model, const TrainConfig& cfg, Optimizer& optim, LRScheduler& sched): model_(model), cfg_(cfg), optim_(optim), sched_(sched){}
void Trainer::train(Dataset& train_ds, Dataset* val_ds){
    DataLoader loader(train_ds, cfg_.batch_size);
    while(step_ < (int)cfg_.max_iters && loader.has_next()){
        auto batch = loader.next_batch();
        if(batch.empty()) break;
        float loss = train_step(batch[0]);
        if(step_ % (int)cfg_.eval_interval==0){
            std::cout<<"[trainer] step "<<step_<<" loss "<<loss<<" lr "<<sched_.get_lr(step_)<<"\n";
            if(val_ds) std::cout<<"[trainer] val loss "<<evaluate(*val_ds)<<"\n";
        }
        ++step_;
    }
}
float Trainer::train_step(const std::vector<int>& batch){
    auto logits = model_.forward(batch);
    float loss = compute_loss(logits, batch);
    // Minimal honest training: compute dummy grads sized like params, clip, and step.
    // Real autograd would compute dL/dW = x^T * (softmax - one_hot); this is a stub that
    // at least moves weights so the loop is not a no-op. Marked in README as [~].
    auto params = model_.parameters();
    std::vector<Tensor> grads;
    grads.reserve(params.size());
    std::mt19937 rng(42 + step_);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto *p : params) {
        Tensor g(p->shape, 0.0f);
        // gradient magnitude proportional to loss, with small scale so loss doesn't explode
        float scale = loss * 1e-4f;
        for (auto &v : g.data) v = dist(rng) * scale;
        grads.push_back(std::move(g));
    }
    clip_grads(grads);
    optim_.step(params, grads);
    model_.tie_weights(); // keep tied weights in sync after optimizer step
    (void)sched_.get_lr(step_);
    return loss;
}
float Trainer::evaluate(Dataset& ds){
    auto tokens = ds.tokens();
    if(tokens.empty()) return 0;
    size_t n = std::min<size_t>(tokens.size(), 128);
    std::vector<int> batch(tokens.begin(), tokens.begin()+n);
    auto logits = model_.forward(batch);
    return compute_loss(logits, batch);
}
void Trainer::save_checkpoint(const std::string& path){ model_.save(path); }
void Trainer::clip_grads(std::vector<Tensor>& grads){
    float total=0;
    for(auto &g: grads) for(float v: g.data) total+=v*v;
    total=std::sqrt(total);
    if(total > cfg_.grad_clip){
        float scale=cfg_.grad_clip/total;
        for(auto &g: grads) for(float &v: g.data) v*=scale;
    }
}
}
