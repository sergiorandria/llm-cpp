#include "llm/trainer.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "llm/loss.h"
#include "llm/utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif
namespace llm {
Trainer::Trainer(GPT& model, const TrainConfig& cfg, Optimizer& optim, LRScheduler& sched)
    : model_(model), cfg_(cfg), optim_(optim), sched_(sched) {
    if (cfg_.deterministic) {
        set_global_seed(42);
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
    }
}
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
    if (cfg_.loss_scale != 1.0f) dlogits = dlogits.scale(cfg_.loss_scale); // A09
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
    if (cfg_.loss_scale != 1.0f) // A09: unscale before clip/optimizer
        for (auto& g : grads) for (auto& v : g.data) v /= cfg_.loss_scale;
    if (has_nonfinite(grads)) { // A07: skip step, keep last good params
        ++skipped_;
        std::cerr << "[trainer] non-finite grad at step " << step_ << " — skipping optimizer step\n";
        return loss;
    }
    clip_grads(grads);
    // D37 metrics
    last_loss_ = loss;
    double n2 = 0;
    for (auto& g : grads) for (float v : g.data) n2 += (double)v * v;
    last_grad_norm_ = (float)std::sqrt(n2);
    float lr = sched_.get_lr(step_);
    optim_.set_lr(lr);
    optim_.step(params, grads);
    model_.tie_weights();
    return loss;
}

// D35 accumulation: sum micro grads, step every K. Loss returned is micro mean.
float Trainer::train_step_accum(const std::vector<int>& micro_batch) {
    size_t K = cfg_.grad_accum_steps < 1 ? 1 : cfg_.grad_accum_steps;
    auto [logits, hidden] = model_.forward_with_hidden(micro_batch);
    float loss = compute_loss(logits, micro_batch);
    model_.zero_grad();
    Tensor dlogits = cross_entropy_backward(logits, micro_batch, cfg_.label_smoothing);
    model_.backward(dlogits, micro_batch, hidden);
    auto params = model_.parameters();
    if (accum_.empty()) {
        accum_.reserve(params.size());
        for (auto* p : params) accum_.emplace_back(p->shape, 0.0f);
    }
    for (size_t i = 0; i < params.size(); ++i) {
        auto* p = params[i];
        if (p->grad.size() != p->data.size()) continue;
        for (size_t j = 0; j < p->data.size(); ++j) accum_[i].data[j] += p->grad[j];
    }
    accum_count_++;
    last_loss_ = loss;
    if (accum_count_ < K) return loss;
    // average, then clip/step like train_step
    std::vector<Tensor> grads = accum_;
    for (auto& g : grads) for (auto& v : g.data) v /= (float)K;
    if (has_nonfinite(grads)) { ++skipped_; accum_count_ = 0; for (auto& a : accum_) a.fill(0); return loss; }
    clip_grads(grads);
    double n2 = 0;
    for (auto& g : grads) for (float v : g.data) n2 += (double)v * v;
    last_grad_norm_ = (float)std::sqrt(n2);
    float lr = sched_.get_lr(step_);
    optim_.set_lr(lr);
    optim_.step(params, grads);
    model_.tie_weights();
    accum_count_ = 0;
    for (auto& a : accum_) a.fill(0);
    return loss;
}

void mean_reduce_grads(const std::vector<std::vector<Tensor>>& shard_grads,
                       std::vector<Tensor>& out) {
    if (shard_grads.empty()) return;
    out = shard_grads[0];
    for (size_t s = 1; s < shard_grads.size(); ++s)
        for (size_t i = 0; i < out.size(); ++i)
            for (size_t j = 0; j < out[i].data.size(); ++j) out[i].data[j] += shard_grads[s][i].data[j];
    float inv = 1.0f / (float)shard_grads.size();
    for (auto& g : out) for (auto& v : g.data) v *= inv;
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
void Trainer::save_train_state(const std::string& prefix) {
    std::filesystem::create_directories(std::filesystem::path(prefix).parent_path().string().empty()
        ? "." : std::filesystem::path(prefix).parent_path().string());
    model_.save_binary(prefix + ".model.bin");
    { std::ofstream os(prefix + ".opt.bin", std::ios::binary); optim_.save_state(os); }
    { std::ofstream js(prefix + ".json"); js << "{\"step\":" << step_ << ",\"skipped\":" << skipped_
        << ",\"seed\":" << global_seed() << "}\n"; }
}
void Trainer::load_train_state(const std::string& prefix) {
    model_.load_binary(prefix + ".model.bin");
    { std::ifstream is(prefix + ".opt.bin", std::ios::binary); if (is) optim_.load_state(is); }
    std::ifstream js(prefix + ".json");
    if (js) {
        std::string s((std::istreambuf_iterator<char>(js)), {});
        auto num = [&](const char* k) {
            auto p = s.find(k); if (p == std::string::npos) return 0;
            return std::stoi(s.substr(s.find(':', p) + 1));
        };
        step_ = num("\"step\""); skipped_ = num("\"skipped\"");
        set_global_seed((uint64_t)num("\"seed\""));
    }
}
bool has_nonfinite(const std::vector<Tensor>& grads) {
    for (auto& g : grads)
        for (float v : g.data)
            if (!std::isfinite(v)) return true;
    return false;
}
void Trainer::clip_grads(std::vector<Tensor>& grads) {
    clip_by_global_norm(grads, cfg_.grad_clip);
}
}  // namespace llm
