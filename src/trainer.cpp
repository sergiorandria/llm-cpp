#include "llm/trainer.h"

#include <cmath>
#include <iostream>

#include "llm/loss.h"
namespace llm {
Trainer::Trainer(GPT& model, const TrainConfig& cfg, Optimizer& optim, LRScheduler& sched)
    : model_(model), cfg_(cfg), optim_(optim), sched_(sched) {}
void Trainer::train(Dataset& train_ds, Dataset* val_ds) {
    DataLoader loader(train_ds, cfg_.batch_size);
    while (step_ < (int)cfg_.max_iters && loader.has_next()) {
        auto batch = loader.next_batch();
        if (batch.empty()) break;
        float loss = train_step(batch[0]);
        if (step_ % (int)cfg_.eval_interval == 0) {
            std::cout << "[trainer] step " << step_ << " loss " << loss << " lr "
                      << sched_.get_lr(step_) << "\n";
            if (val_ds) std::cout << "[trainer] val loss " << evaluate(*val_ds) << "\n";
        }
        ++step_;
    }
}
float Trainer::train_step(const std::vector<int>& batch) {
    auto [logits, hidden] = model_.forward_with_hidden(batch);
    float loss = compute_loss(logits, batch);
    // Real backward through entire graph (or gated fallback)
    if (cfg_.allow_untrained_params) {
        // Legacy random-noise path (explicitly gated)
        auto params = model_.parameters();
        std::vector<Tensor> grads;
        grads.reserve(params.size());
        Tensor grad_lm_head({hidden.shape[1], logits.shape[1]}, 0.0f);
        {
            size_t T = logits.shape[0], V = logits.shape[1], C = hidden.shape[1];
            Tensor probs = logits.softmax(1);
            for (size_t i = 0; i < T && i < batch.size(); ++i) {
                int tgt = batch[i];
                if (tgt >= 0 && (size_t)tgt < V) probs(i, tgt) -= 1.0f;
                for (size_t j = 0; j < V; ++j) probs(i, j) /= float(T);
            }
            Tensor g({C, V}, 0.0f);
            for (size_t c = 0; c < C; ++c)
                for (size_t v = 0; v < V; ++v) {
                    float acc = 0;
                    for (size_t t = 0; t < T; ++t) acc += hidden(t, c) * probs(t, v);
                    g(c, v) = acc;
                }
            grad_lm_head = g;
        }
        std::mt19937 rng(42 + step_);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < params.size(); ++i) {
            auto* p = params[i];
            Tensor g(p->shape, 0.0f);
            if (p->shape == grad_lm_head.shape)
                g = grad_lm_head;
            else if (p->shape.size() == 2 && grad_lm_head.shape.size() == 2 &&
                     p->shape[0] == grad_lm_head.shape[1] && p->shape[1] == grad_lm_head.shape[0]) {
                for (size_t a = 0; a < p->shape[0]; ++a)
                    for (size_t b = 0; b < p->shape[1]; ++b) g(a, b) = grad_lm_head(b, a);
            } else {
                float scale = loss * 1e-4f;
                for (auto& v : g.data) v = dist(rng) * scale;
            }
            grads.push_back(std::move(g));
        }
        clip_grads(grads);
        float lr = sched_.get_lr(step_);
        optim_.set_lr(lr);
        optim_.step(params, grads);
        model_.tie_weights();
        return loss;
    }
    // Real backward: zero grads, compute dlogits, backprop through entire model
    model_.zero_grad();
    Tensor dlogits = cross_entropy_backward(logits, batch, cfg_.label_smoothing);
    model_.backward(dlogits, batch, hidden);
    // Collect grads from parameters' grad fields
    auto params = model_.parameters();
    std::vector<Tensor> grads;
    grads.reserve(params.size());
    for (auto* p : params) {
        Tensor g(p->shape, 0.0f);
        if (p->grad.size() == p->data.size()) g.data = p->grad;
        // else stays zero (should have been set by backward)
        grads.push_back(std::move(g));
    }
    clip_grads(grads);
    float lr = sched_.get_lr(step_);
    optim_.set_lr(lr);
    optim_.step(params, grads);
    model_.tie_weights();
    return loss;
}
float Trainer::evaluate(Dataset& ds) {
    auto tokens = ds.tokens();
    if (tokens.empty()) return 0;
    size_t n = std::min<size_t>(tokens.size(), 128);
    std::vector<int> batch(tokens.begin(), tokens.begin() + n);
    auto logits = model_.forward(batch);
    return compute_loss(logits, batch);
}
void Trainer::save_checkpoint(const std::string& path) {
    model_.save(path);
}
void Trainer::clip_grads(std::vector<Tensor>& grads) {
    clip_by_global_norm(grads, cfg_.grad_clip);
}
}  // namespace llm
