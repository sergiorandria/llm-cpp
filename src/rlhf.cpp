#include "llm/rlhf.h"

#include <cmath>
#include <numeric>

#include "llm/loss.h"
#include "llm/optimizer.h"

namespace llm {

RewardModel::RewardModel(const GPT& base) : base_(base) {
    size_t n_embd = base_.config().n_embd;
    head_w_ = Tensor({n_embd}, 0.0f);
    head_w_.randn(0, 0.02f);
    head_b_ = 0.0f;
}

Tensor RewardModel::get_pooled_hidden(const std::vector<int>& prompt, const std::vector<int>& completion) const {
    std::vector<int> seq = prompt;
    seq.insert(seq.end(), completion.begin(), completion.end());
    if(seq.empty()) return Tensor({head_w_.shape[0]}, 0.0f);
    // Truncate to block_size for forward
    const auto& cfg = base_.config();
    if(seq.size() > cfg.block_size){
        // Use last block_size tokens for context (like generate)
        seq.assign(seq.end() - cfg.block_size, seq.end());
    }
    auto [logits, hidden] = base_.forward_with_hidden(seq);
    size_t T = hidden.shape[0];
    size_t C = hidden.shape[1];
    // Pool over completion part: positions corresponding to completion tokens
    // After truncation, prompt may be partially truncated; approximate by pooling last completion.size() rows
    size_t comp_len = completion.size();
    if(comp_len==0) comp_len=1;
    // hidden rows correspond to seq; pool last comp_len rows (or all if seq smaller)
    size_t pool_start = T > comp_len ? T - comp_len : 0;
    size_t pool_n = T - pool_start;
    Tensor pooled({C}, 0.0f);
    for(size_t j=0;j<C;++j){
        float sum=0;
        for(size_t i=pool_start;i<T;++i) sum += hidden(i,j);
        pooled.data[j] = sum / float(pool_n);
    }
    return pooled;
}

float RewardModel::score(const std::vector<int>& prompt, const std::vector<int>& completion) const {
    if(prompt.empty() && completion.empty()) return head_b_;
    Tensor pooled = get_pooled_hidden(prompt, completion);
    float r = head_b_;
    for(size_t i=0;i<pooled.data.size() && i<head_w_.data.size();++i) r += pooled.data[i]*head_w_.data[i];
    return r;
}

void RewardModel::train_step(const std::vector<int>& prompt, const std::vector<int>& chosen,
                    const std::vector<int>& rejected, float lr){
    Tensor hc = get_pooled_hidden(prompt, chosen);
    Tensor hr = get_pooled_hidden(prompt, rejected);
    float rc = head_b_;
    float rr = head_b_;
    for(size_t i=0;i<hc.data.size() && i<head_w_.data.size();++i){
        rc += hc.data[i]*head_w_.data[i];
        rr += hr.data[i]*head_w_.data[i];
    }
    float delta = rc - rr;
    // Sigmoid
    float sigmoid = 1.0f/(1.0f+std::exp(-delta));
    // Loss = -log(sigmoid(delta)); dL/d delta = -(1 - sigmoid)
    float dL_ddelta = -(1.0f - sigmoid);
    // Grad for head_w: dL/dW = dL/ddelta * (hc - hr)
    for(size_t i=0;i<head_w_.data.size();++i){
        float hc_v = i<hc.data.size()? hc.data[i]:0;
        float hr_v = i<hr.data.size()? hr.data[i]:0;
        float grad = dL_ddelta * (hc_v - hr_v);
        head_w_.data[i] -= lr * grad;
    }
    // Bias cancels in delta, so no update (would be dL/db = dL/ddelta * (1-1)=0)
    // For completeness, we could add small bias grad if we used separate biases, but keep 0
}

void ppo_step(GPT& policy, const RewardModel& reward, const std::vector<int>& prompt) {
    RewardConfig cfg;
    ppo_step(policy, reward, prompt, cfg);
}

void ppo_step(GPT& policy, const RewardModel& reward, const std::vector<int>& prompt, const RewardConfig& cfg) {
    if (prompt.empty()) return;
    // Generate completion
    auto completion = policy.generate(prompt, 4, 0.8f, 0, 1.0f);
    std::vector<int> new_tokens;
    if (completion.size() > prompt.size()) {
        new_tokens.assign(completion.begin() + prompt.size(), completion.end());
    } else {
        new_tokens = {1, 2, 3};
    }
    float r = reward.score(prompt, new_tokens);
    r = std::max(-5.0f, std::min(5.0f, r));
    // Advantage with running baseline (no value head)
    static float baseline = 0.0f;
    float advantage = r - baseline;
    baseline = 0.9f * baseline + 0.1f * r;

    std::vector<int> seq = prompt;
    seq.insert(seq.end(), new_tokens.begin(), new_tokens.end());

    // Old policy snapshot (reference for ratio and KL)
    Tensor old_logits = policy.forward(seq);
    Tensor old_probs = old_logits.softmax(1);

    policy.zero_grad();
    auto [logits, hidden] = policy.forward_with_hidden(seq);
    size_t T = logits.shape[0], V = logits.shape[1];
    Tensor new_probs = logits.softmax(1);

    // Compute PPO clipped objective + KL penalty
    Tensor dlogits({T, V}, 0.0f);
    float eps = cfg.ppo_clip_eps;
    float kl_coef = cfg.kl_coef;
    for (size_t i = 0; i < T && i < seq.size(); ++i) {
        int tgt = seq[i];
        tgt = ((tgt % (int)V) + (int)V) % (int)V;
        float old_p = std::max(old_probs(i, tgt), 1e-8f);
        float new_p = std::max(new_probs(i, tgt), 1e-8f);
        float ratio = new_p / old_p;
        float clipped = std::max(1.0f - eps, std::min(1.0f + eps, ratio));
        // PPO clipped objective: use_clipped when ratio is outside [1-eps,1+eps] and moving further would improve objective
        bool use_clipped = false;
        if ((advantage > 0 && ratio > 1.0f + eps) || (advantage < 0 && ratio < 1.0f - eps)) {
            use_clipped = true;
        } else {
            // Also check min logic: if clipped objective is smaller, it is selected
            float obj = ratio * advantage;
            float obj_clipped = clipped * advantage;
            if (std::min(obj, obj_clipped) != obj) use_clipped = true;
        }
        if (use_clipped) {
            // Gradient is zero when clipped (no incentive to move further)
            for (size_t j = 0; j < V; ++j) dlogits(i, j) = 0.0f;
        } else {
            // Gradient of ratio*adv wrt logits: adv * ratio * (one_hot - new_prob)
            for (size_t j = 0; j < V; ++j) {
                float grad = new_probs(i, j);
                if ((int)j == tgt) grad -= 1.0f;
                // dlog ratio = (one_hot - new_prob)
                // d objective = adv * ratio * (one_hot - new_prob)
                // loss = -objective, so grad = -adv * ratio * (one_hot - new_prob) = adv * ratio * (new_prob - one_hot)
                float g = advantage * ratio * grad;
                g /= float(T);
                // KL penalty: add kl_coef * (new_prob - old_prob)  (pushes towards old)
                float kl_grad = kl_coef * (new_probs(i, j) - old_probs(i, j)) / float(T);
                g += kl_grad;
                // For loss we want negative, but we already did (prob - one_hot) = - (one_hot - prob), so g is loss grad
                // We need dlogits for backward which expects gradient of loss wrt logits (which is (prob - one_hot)*adv*ratio ...)
                // Our g above is already loss grad (since grad = adv*ratio*(prob - one_hot)), so use directly
                // But earlier simple REINFORCE used -r*(one_hot - prob) = r*(prob - one_hot), same sign, so consistent
                dlogits(i, j) = g;
            }
        }
        // If not clipped but we still need to handle KL for other tokens? Already added
        if (use_clipped) {
            // Even when clipped, still apply KL penalty gradient
            for (size_t j = 0; j < V; ++j) {
                float kl_grad = kl_coef * (new_probs(i, j) - old_probs(i, j)) / float(T);
                dlogits(i, j) = kl_grad;
            }
        }
    }
    policy.backward(dlogits, seq, hidden);
    // Use shared optimizer (Adam) instead of hand-rolled SGD
    // Collect grads
    auto params = policy.parameters();
    std::vector<Tensor> grads;
    grads.reserve(params.size());
    for (auto* p : params) {
        Tensor g(p->shape, 0.0f);
        if (p->grad.size() == p->data.size()) {
            for (size_t i = 0; i < g.data.size(); ++i) g.data[i] = p->grad[i];
        }
        grads.push_back(std::move(g));
    }
    // Use Adam with lr 1e-3
    static Adam adam_opt(1e-3f);
    adam_opt.step(params, grads);
    // Also need to zero grads after step for next call
    policy.zero_grad();
}

}  // namespace llm
