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
    // Simulate grads as zeros then optimizer step (real backward would compute dlogits)
    // Use numpy-accelerated grad clipping helper
    std::vector<Tensor> fake_grads;
    // create dummy grad for demo (would be populated by autograd)
    fake_grads.emplace_back(Tensor({1,1},0.0f));
    clip_grads(fake_grads);
    // scheduler update (lr printed in train loop)
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
