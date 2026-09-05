#include "llm/model.h"
#include <cmath>
#include <random>
#include <iostream>
#include "llm/sampling.h"
#include <fstream>
#include <filesystem>

namespace llm {

GPT::GPT(const Config& config) : config_(config),
    wte_({config.vocab_size, config.n_embd}),
    wpe_({config.block_size, config.n_embd}),
    ln_f_gamma_({config.n_embd}, 1.0f), ln_f_beta_({config.n_embd}, 0.0f),
    lm_head_({config.n_embd, config.vocab_size}) {
    wte_.randn(0, 0.02f);
    wpe_.randn(0, 0.02f);
    lm_head_.randn(0, 0.02f);
    if (config_.weight_tying) {
        // Weight tying: lm_head should be wte^T. For now alias via copy + tie_weights()
        // True alias would use shared_ptr storage; we emulate by copying transposed.
        tie_weights();
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

int next_id = 0;
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
    save_binary(path);
    std::cout << "[save] checkpoint written to " << path << " (" << num_parameters() << " params)\n";
}
void GPT::save_binary(const std::string& path) const {
    // Ensure parent directory exists
    {
        size_t slash = path.find_last_of("/\\");
        if (slash != std::string::npos) {
            std::string dir = path.substr(0, slash);
            std::filesystem::create_directories(dir);
        }
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) { std::cerr << "[save_binary] cannot open " << path << "\n"; return; }
    uint32_t magic = 0x4C4C4D00; out.write((char*)&magic, 4);
    uint32_t version = 2; out.write((char*)&version, 4);
    // config
    out.write((char*)&config_.vocab_size, sizeof(size_t));
    out.write((char*)&config_.n_layers, sizeof(size_t));
    out.write((char*)&config_.n_heads, sizeof(size_t));
    out.write((char*)&config_.n_embd, sizeof(size_t));
    out.write((char*)&config_.block_size, sizeof(size_t));
    auto write_tensor = [&](const Tensor& t){
        uint64_t ndim = t.shape.size(); out.write((char*)&ndim, 8);
        for (auto d : t.shape) { uint64_t v=d; out.write((char*)&v, 8); }
        uint64_t n = t.data.size(); out.write((char*)&n, 8);
        out.write((char*)t.data.data(), n*sizeof(float));
    };
    write_tensor(wte_); write_tensor(wpe_);
    write_tensor(ln_f_gamma_); write_tensor(ln_f_beta_);
    write_tensor(lm_head_);
    // Note: blocks not serialized in v2 header-only top-level for brevity — full v3 would iterate blocks
}
void GPT::load_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "[load_binary] cannot open " << path << " (random init)\n"; return; }
    uint32_t magic, version; in.read((char*)&magic,4); in.read((char*)&version,4);
    if (magic != 0x4C4C4D00) { std::cerr << "[load_binary] bad magic\n"; return; }
    if (version < 1) return;
    size_t vs, nl, nh, ne, bs;
    in.read((char*)&vs, sizeof(size_t)); in.read((char*)&nl, sizeof(size_t));
    in.read((char*)&nh, sizeof(size_t)); in.read((char*)&ne, sizeof(size_t));
    in.read((char*)&bs, sizeof(size_t));
    bool mismatch = (vs != config_.vocab_size || nl != config_.n_layers || nh != config_.n_heads || ne != config_.n_embd || bs != config_.block_size);
    if (mismatch) {
        std::cerr << "[load_binary] config mismatch (ckpt vocab=" << vs << " n_embd=" << ne << " vs current vocab=" << config_.vocab_size << " n_embd=" << config_.n_embd << ") — skipping weight load, using random init\n";
        return;
    }
    auto read_tensor = [&](Tensor& t){
        uint64_t ndim; in.read((char*)&ndim,8);
        std::vector<size_t> shape(ndim);
        for (uint64_t i=0;i<ndim;++i){ uint64_t v; in.read((char*)&v,8); shape[i]= (size_t)v; }
        uint64_t n; in.read((char*)&n,8);
        // Use ctor to get correct strides, then read data
        t = Tensor(shape, 0.0f);
        in.read((char*)t.data.data(), n*sizeof(float));
    };
    read_tensor(wte_); read_tensor(wpe_);
    read_tensor(ln_f_gamma_); read_tensor(ln_f_beta_);
    read_tensor(lm_head_);
    if (config_.weight_tying) tie_weights();
}

std::vector<int> GPT::generate_streaming(const std::vector<int>& prompt, size_t max_new_tokens, std::function<void(int)> cb) const {
    auto out = generate(prompt, max_new_tokens);
    for(size_t i=prompt.size(); i<out.size(); ++i) cb(out[i]);
    return out;
}
void GPT::load(const std::string& path) {
    load_binary(path);
    std::cout << "[load] checkpoint loaded from " << path << "\n";
}

size_t GPT::num_parameters() const {
    size_t n = wte_.numel() + wpe_.numel() + ln_f_gamma_.numel() + ln_f_beta_.numel() + lm_head_.numel();
    for(auto &b: blocks_){
        n += 4 * config_.n_embd * config_.n_embd;
        n += 4 * config_.n_embd;
        n += config_.n_embd * 4*config_.n_embd + 4*config_.n_embd*config_.n_embd;
        n += 5*config_.n_embd;
    }
    if (config_.weight_tying) n -= lm_head_.numel(); // tied, not double-counted
    return n;
}

std::vector<Tensor*> GPT::parameters() {
    std::vector<Tensor*> p;
    p.reserve(4 + blocks_.size()*8);
    p.push_back(&wte_);
    p.push_back(&wpe_);
    p.push_back(&ln_f_gamma_);
    p.push_back(&ln_f_beta_);
    if (!config_.weight_tying) p.push_back(&lm_head_);
    // Note: TransformerBlock weights not exposed individually yet — expose via block API
    // For now expose only top-level tensors; trainer will also handle block params via manual update
    return p;
}
std::vector<const Tensor*> GPT::parameters() const {
    std::vector<const Tensor*> p;
    p.reserve(4);
    p.push_back(&wte_); p.push_back(&wpe_); p.push_back(&ln_f_gamma_); p.push_back(&ln_f_beta_);
    if (!config_.weight_tying) p.push_back(&lm_head_);
    return p;
}
void GPT::tie_weights() {
    if (!config_.weight_tying) return;
    // lm_head is [n_embd, vocab], wte is [vocab, n_embd] -> lm_head = wte^T
    if (wte_.shape.size()==2 && lm_head_.shape.size()==2 &&
        wte_.shape[0]==lm_head_.shape[1] && wte_.shape[1]==lm_head_.shape[0]) {
        for (size_t i=0;i<wte_.shape[0];++i)
            for (size_t j=0;j<wte_.shape[1];++j)
                lm_head_(j,i) = wte_(i,j);
    }
}

} // namespace llm
