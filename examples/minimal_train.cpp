#include "llm/model.h"
#include "llm/dataset.h"
#include "llm/trainer.h"
#include "llm/optimizer.h"
#include "llm/scheduler.h"
int main(){
    llm::Config cfg; cfg.vocab_size=256; cfg.n_layers=2; cfg.n_heads=4; cfg.n_embd=64; cfg.block_size=32;
    llm::GPT model(cfg);
    llm::Dataset ds("data/input.txt", 32);
    llm::AdamW optim(6e-4f);
    llm::CosineScheduler sched(6e-4f, 100, 5000);
    llm::TrainConfig tcfg;
    llm::Trainer trainer(model, tcfg, optim, sched);
    trainer.train(ds);
    return 0;
}
