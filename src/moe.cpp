#include "llm/moe.h"

#include <algorithm>
#include <cmath>

namespace llm {

MoEFFN::MoEFFN(size_t n_embd, MoEConfig cfg) : cfg_(cfg), gate_({n_embd, cfg.n_experts}) {
    size_t hidden = 4 * n_embd;
    gate_.randn(0, 0.02f);
    experts_W1_.reserve(cfg_.n_experts);
    experts_W2_.reserve(cfg_.n_experts);
    for (size_t e = 0; e < cfg_.n_experts; ++e) {
        Tensor w1({n_embd, hidden});
        Tensor w2({hidden, n_embd});
        w1.randn(0, 0.02f);
        w2.randn(0, 0.02f);
        experts_W1_.push_back(std::move(w1));
        experts_W2_.push_back(std::move(w2));
    }
}

Tensor MoEFFN::forward(const Tensor& x) const {
    // x: [T, n_embd]
    size_t T = x.shape[0];
    size_t C = x.shape[1];
    size_t E = cfg_.n_experts;
    size_t topk = cfg_.top_k;
    // Compute gate scores: x * gate => [T, E]
    Tensor gate_scores = x.matmul(gate_);  // [T, E]
    // Softmax per token to get routing probs
    Tensor gate_probs = gate_scores.softmax(1);  // [T, E]
    Tensor out({T, C}, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        // Find top-k experts for token t
        std::vector<std::pair<float, size_t>> scored;
        scored.reserve(E);
        for (size_t e = 0; e < E; ++e) scored.emplace_back(gate_probs(t, e), e);
        std::partial_sort(scored.begin(), scored.begin() + topk, scored.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        // Renormalize topk probs
        float sum = 0;
        for (size_t k = 0; k < topk; ++k) sum += scored[k].first;
        if (sum < 1e-8f) sum = 1.0f;
        for (size_t k = 0; k < topk; ++k) {
            size_t e = scored[k].second;
            float w = scored[k].first / sum;
            // expert forward: x[t] * W1 -> gelu -> * W2
            Tensor xt({1, C}, 0.0f);
            for (size_t c = 0; c < C; ++c) xt(0, c) = x(t, c);
            Tensor h = xt.matmul(experts_W1_[e]);  // [1, hidden]
            // GELU
            for (auto& v : h.data) {
                v = 0.5f * v *
                    (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265f) * (v + 0.044715f * v * v * v)));
            }
            Tensor y = h.matmul(experts_W2_[e]);  // [1, C]
            for (size_t c = 0; c < C; ++c) out(t, c) += w * y(0, c);
        }
    }
    return out;
}

}  // namespace llm
