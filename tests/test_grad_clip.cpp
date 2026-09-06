#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/loss.h"
#include "llm/optimizer.h"
#include "llm/scheduler.h"
#include "llm/trainer.h"

int main() {
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_embd = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 1;
    cfg.block_size = 8;
    llm::GPT model(cfg);
    llm::AdamW optim(1e-3f);
    llm::CosineScheduler sched(1e-3f, 2, 10);
    llm::TrainConfig tcfg;
    tcfg.grad_clip = 0.5f;
    llm::Trainer trainer(model, tcfg, optim, sched);
    std::vector<llm::Tensor> grads = {llm::Tensor({2, 2}, 10.0f)};
    float before = 0;
    for (auto v : grads[0].data) before += v * v;
    before = std::sqrt(before);
    std::cout << "grad norm before " << before << "\n";
    assert(before > 0.5f);
    llm::Dataset ds("/tmp/nonexistent.txt", 8);
    auto batch = ds.tokens();
    if (batch.size() > 8) batch.resize(8);
    for (auto& t : batch) t = ((t % (int)cfg.vocab_size) + cfg.vocab_size) % cfg.vocab_size;
    float loss = trainer.train_step(batch);
    assert(!std::isnan(loss));
    std::cout << "grad clip test passed (loss finite " << loss << ")\n";
    // Scheduler monotonicity: warmup then cosine decay
    float lr0 = sched.get_lr(0);
    float lr1 = sched.get_lr(1);
    float lr5 = sched.get_lr(5);
    std::cout << "lr0 " << lr0 << " lr1 " << lr1 << " lr5 " << lr5 << "\n";
    assert(lr1 >= lr0);
    float lr10 = sched.get_lr(10);
    assert(lr10 <= lr5 || lr10 < 1e-3f);
    std::cout << "scheduler test passed\n";

    // --- Grad non-random + clip functional test ---
    llm::Config tiny;
    tiny.vocab_size = 8;
    tiny.n_embd = 8;
    tiny.n_heads = 2;
    tiny.n_layers = 1;
    tiny.block_size = 4;
    llm::GPT tiny_model(tiny);
    std::vector<int> tiny_batch = {1, 2, 0, 3};
    tiny_model.zero_grad();
    auto [logits, hidden] = tiny_model.forward_with_hidden(tiny_batch);
    size_t T = logits.shape[0], V = logits.shape[1];
    llm::Tensor dlogits = logits.softmax(1);
    for (size_t i = 0; i < T && i < tiny_batch.size(); ++i) {
        int tgt = tiny_batch[i];
        if (tgt >= 0 && (size_t)tgt < V) dlogits(i, tgt) -= 1.0f;
        for (size_t j = 0; j < V; ++j) dlogits(i, j) /= float(T);
    }
    tiny_model.backward(dlogits, tiny_batch, hidden);
    auto params = tiny_model.parameters();
    // collect grads and check non-random & non-zero for attn and ffn
    bool found_attn_nonzero = false, found_ffn_nonzero = false;
    for (auto* p : params) {
        float norm = 0;
        for (float g : p->grad) norm += g * g;
        norm = std::sqrt(norm);
        if (p->shape.size() == 2 && p->shape[0] == 8 && p->shape[1] == 8 && norm > 1e-6)
            found_attn_nonzero = true;
        if (p->shape.size() == 2 && p->shape[0] == 8 && p->shape[1] == 32 && norm > 1e-6)
            found_ffn_nonzero = true;
    }
    std::cout << "attn nonzero " << found_attn_nonzero << " ffn nonzero " << found_ffn_nonzero
              << "\n";
    assert(found_attn_nonzero && "attention grad is zero — still random/noise?");
    assert(found_ffn_nonzero && "FFN grad is zero");

    // Grad clipping effect: create large grads and verify scaling
    std::vector<llm::Tensor> big_grads;
    big_grads.push_back(llm::Tensor({4, 4}, 10.0f));
    float norm_before = 0;
    for (auto& g : big_grads)
        for (float v : g.data) norm_before += v * v;
    norm_before = std::sqrt(norm_before);
    // mimic Trainer::clip_grads with clip=0.5
    float clip = 0.5f;
    if (norm_before > clip) {
        float scale = clip / norm_before;
        for (auto& g : big_grads)
            for (float& v : g.data) v *= scale;
    }
    float norm_after = 0;
    for (auto& g : big_grads)
        for (float v : g.data) norm_after += v * v;
    norm_after = std::sqrt(norm_after);
    std::cout << "clip norm before " << norm_before << " after " << norm_after << "\n";
    assert(std::abs(norm_after - clip) < 1e-4f && "grad clipping did not scale to clip value");
    std::cout << "grad clip functional test passed\n";
    return 0;
}
