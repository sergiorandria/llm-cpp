#include "llm/model.h"
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>

namespace llm {

GPT::GPT(const Config& config) : config_(config),
    wte_({config.vocab_size, config.n_embd}),
    wpe_({config.block_size, config.n_embd}),
    ln_f_gamma_({config.n_embd}, 1.0f), ln_f_beta_({config.n_embd}, 0.0f),
    lm_head_({config.n_embd, config.vocab_size}) {
    wte_.randn(0, 0.02f);
    wpe_.randn(0, 0.02f);
    lm_head_.randn(0, 0.02f);
    blocks_.reserve(config.n_layers);
    for (size_t i = 0; i < config.n_layers; ++i) {
        blocks_.emplace_back(config.n_embd, config.n_heads, config.block_size);
    }
}

static Tensor rope(const Tensor& x, size_t seq_len){
    // Stub RoPE: return x unchanged (real impl rotates pairs)
    (void)seq_len;
    return x;
}
Tensor GPT::forward(const std::vector<int>& tokens) const {
    size_t T = tokens.size();
    assert(T <= config_.block_size);
    Tensor x({T, config_.n_embd}, 0.0f);
    // x = wte[tokens] + wpe[pos]
    for (size_t t = 0; t < T; ++t) {
        int tok = tokens[t];
        for (size_t j = 0; j < config_.n_embd; ++j) {
            x(t, j) = wte_(tok, j) + wpe_(t, j);
        }
    }
    for (auto& block : blocks_) {
        x = block.forward(x);
    }
    x = x.layernorm();
    Tensor logits = x.matmul(lm_head_); // [T, vocab]
    return logits;
}

std::vector<int> GPT::generate(const std::vector<int>& prompt, size_t max_new_tokens,
                               float temperature, int top_k) const {
    std::vector<int> out = prompt;
    std::mt19937 rng(42);
    for (size_t step = 0; step < max_new_tokens; ++step) {
        size_t start = out.size() > config_.block_size ? out.size() - config_.block_size : 0;
        std::vector<int> ctx(out.begin() + start, out.end());
        Tensor logits = forward(ctx);
        // take last token logits
        size_t T = logits.shape[0];
        std::vector<float> last_logits(config_.vocab_size);
        for (size_t j = 0; j < config_.vocab_size; ++j) last_logits[j] = logits(T-1, j);

        // temperature
        if (temperature != 1.0f) {
            for (auto& v : last_logits) v /= temperature;
        }

        // top-k (simple)
        int next_id = 0;
        if (top_k > 0) {
            // naive: find top_k indices, sample within them
            // for skeleton, just greedy within top-k after sorting
            std::vector<int> idx(config_.vocab_size);
            for (size_t i=0;i<idx.size();++i) idx[i]=i;
            std::partial_sort(idx.begin(), idx.begin()+top_k, idx.end(),
                [&](int a, int b){ return last_logits[a] > last_logits[b]; });
            // sample uniformly among top_k for demo
            std::uniform_int_distribution<int> dist(0, top_k-1);
            next_id = idx[dist(rng)];
        } else {
            float maxv = last_logits[0];
            for (size_t i=1;i<last_logits.size();++i) maxv = std::max(maxv, last_logits[i]);
            float sum = 0;
            for (auto& v : last_logits) { v = std::exp(v - maxv); sum += v; }
            for (auto& v : last_logits) v /= sum;
            std::discrete_distribution<int> dist(last_logits.begin(), last_logits.end());
            next_id = dist(rng);
        }
        out.push_back(next_id);
    }
    return out;
}

void GPT::save(const std::string& path) const {
    std::cout << "[save] would write checkpoint to " << path << " (" << num_parameters() << " params)\n";
}

void GPT::load(const std::string& path) {
    std::cout << "[load] would load checkpoint from " << path << "\n";
}

size_t GPT::num_parameters() const {
    size_t n = wte_.numel() + wpe_.numel() + ln_f_gamma_.numel() + lm_head_.numel();
    // rough: each block ~ 12 * n_embd^2
    n += blocks_.size() * 12 * config_.n_embd * config_.n_embd;
    return n;
}

} // namespace llm
