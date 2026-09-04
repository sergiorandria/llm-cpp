#include "llm/model.h"
#include <cmath>
#include <random>
#include <iostream>
#include "llm/sampling.h"
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
if(config_.weight_tying && wte_.shape == lm_head_.shape){
    // tie: share storage (copy for stub, real would alias)
    lm_head_.data = wte_.data;
}
if(config_.pos_encoding == PosEncoding::Sinusoidal){
    // fill wpe with sinusoidal
    for(size_t pos=0; pos<config_.block_size; ++pos)
        for(size_t i=0;i<config_.n_embd;++i){
            float angle = pos / std::pow(10000.0f, 2*(i/2)/(float)config_.n_embd);
            wpe_(pos,i) = (i%2==0) ? std::sin(angle) : std::cos(angle);
        }
}
    blocks_.reserve(config.n_layers);
    for (size_t i = 0; i < config.n_layers; ++i) {
        blocks_.emplace_back(config.n_embd, config.n_heads, config.block_size);
    }
}

static Tensor rope(const Tensor& x, size_t seq_len){
    Tensor out = x;
    // RoPE: rotate pairs (d/2) by angle = pos / 10000^(2i/d)
    for(size_t pos=0; pos<seq_len && pos < x.shape[0]; ++pos){
        for(size_t i=0; i+1 < x.shape[1]; i+=2){
            float angle = pos / std::pow(10000.0f, (float)i / x.shape[1]);
            float cos_a = std::cos(angle), sin_a = std::sin(angle);
            float x0 = x(pos,i), x1 = x(pos,i+1);
            out(pos,i) = x0 * cos_a - x1 * sin_a;
            out(pos,i+1) = x0 * sin_a + x1 * cos_a;
        }
    }
    return out;
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
    if(config_.pos_encoding == PosEncoding::RoPE) x = rope(x, T);
    for (auto& block : blocks_) {
        x = block.forward(x);
    }
    x = x.layernorm();
    Tensor logits = x.matmul(lm_head_); // [T, vocab]
    return logits;
}

std::vector<int> GPT::generate(const std::vector<int>& prompt, size_t max_new_tokens,
                               float temperature, int top_k, float top_p, float rep_penalty) const {
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

// repetition penalty
if(rep_penalty != 1.0f) last_logits = apply_repetition_penalty(last_logits, out, rep_penalty);
// temperature
if (temperature != 1.0f) {
    for (auto& v : last_logits) v /= temperature;
}

// top-p handling after temperature
if(top_p < 1.0f && top_k==0){
    next_id = sample_top_p(last_logits, top_p, 1.0f);
} else if (top_k > 0) {
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

std::vector<int> GPT::generate_streaming(const std::vector<int>& prompt, size_t max_new_tokens, std::function<void(int)> cb) const {
    auto out = generate(prompt, max_new_tokens);
    for(size_t i=prompt.size(); i<out.size(); ++i) cb(out[i]);
    return out;
}
void GPT::load(const std::string& path) {
    std::cout << "[load] would load checkpoint from " << path << "\n";
}

size_t GPT::num_parameters() const {
    size_t n = wte_.numel() + wpe_.numel() + ln_f_gamma_.numel() + ln_f_beta_.numel() + lm_head_.numel();
    for(auto &b: blocks_){
        // attn: 4* C*C + 4*C biases, ffn: 2* C*4C + 4C + C etc.
        n += 4 * config_.n_embd * config_.n_embd; // Wq,Wk,Wv,Wo
        n += 4 * config_.n_embd; // biases q,k,v,o (if bias)
        n += config_.n_embd * 4*config_.n_embd + 4*config_.n_embd*config_.n_embd; // ffn W1,W2
        n += 5*config_.n_embd; // layernorm gammas/betas
    }
    return n;
}

} // namespace llm
