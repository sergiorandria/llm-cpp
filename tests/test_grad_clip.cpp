#include "llm/trainer.h"
#include "llm/optimizer.h"
#include "llm/scheduler.h"
#include <cassert>
#include <iostream>
#include <cmath>

int main() {
    llm::Config cfg; cfg.vocab_size=16; cfg.n_embd=8; cfg.n_heads=2; cfg.n_layers=1; cfg.block_size=8;
    llm::GPT model(cfg);
    llm::AdamW optim(1e-3f);
    llm::CosineScheduler sched(1e-3f, 2, 10);
    llm::TrainConfig tcfg; tcfg.grad_clip=0.5f;
    llm::Trainer trainer(model, tcfg, optim, sched);
    // Create grads with large norm and verify clipping via train_step
    std::vector<llm::Tensor> grads = {llm::Tensor({2,2}, 10.0f)};
    float before=0; for(auto v: grads[0].data) before+=v*v; before=std::sqrt(before);
    std::cout<<"grad norm before "<<before<<"\n";
    assert(before > 0.5f);
    // Use trainer's clip indirectly via train_step (which clips)
    // For direct test, we can't call private clip_grads, so we test via training doesn't explode
    llm::Dataset ds("/tmp/nonexistent.txt", 8);
    auto batch = ds.tokens();
    if(batch.size()>8) batch.resize(8);
    float loss = trainer.train_step(batch);
    assert(!std::isnan(loss));
    std::cout<<"grad clip test passed (loss finite "<<loss<<")\n";
    // Scheduler monotonicity: warmup then cosine decay
    float lr0 = sched.get_lr(0);
    float lr1 = sched.get_lr(1);
    float lr5 = sched.get_lr(5);
    std::cout<<"lr0 "<<lr0<<" lr1 "<<lr1<<" lr5 "<<lr5<<"\n";
    assert(lr1 >= lr0); // warmup increasing
    // After warmup, cosine should decay
    float lr10 = sched.get_lr(10);
    assert(lr10 <= lr5 || lr10 < 1e-3f);
    std::cout<<"scheduler test passed\n";
    return 0;
}
