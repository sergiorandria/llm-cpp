#include "llm/model.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "llm/kv_cache.h"
#include "llm/profiling.h"
#include "llm/sampling.h"
namespace llm {

GPT::GPT(const Config& config)
    : config_(config),
      wte_({config.vocab_size, config.n_embd}),
      wpe_({config.block_size, config.n_embd}),
      ln_f_gamma_({config.n_embd}, 1.0f),
      ln_f_beta_({config.n_embd}, 0.0f),
      lm_head_({config.n_embd, config.vocab_size}) {
    wte_.randn(0, 0.02f);
    wpe_.randn(0, 0.02f);
    lm_head_.randn(0, 0.02f);
    if (config_.weight_tying) {
        // Weight tying: lm_head should be wte^T. For now alias via copy + tie_weights()
        // True alias would use shared_ptr storage; we emulate by copying transposed.
        tie_weights();
    }
    if (config_.pos_encoding == PosEncoding::Sinusoidal) {
        // fill wpe with sinusoidal
        for (size_t pos = 0; pos < config_.block_size; ++pos)
            for (size_t i = 0; i < config_.n_embd; ++i) {
                float angle = pos / std::pow(10000.0f, 2 * (i / 2) / (float)config_.n_embd);
                wpe_(pos, i) = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
            }
    }
    blocks_.reserve(config.n_layers);
    for (size_t i = 0; i < config.n_layers; ++i) {
        TransformerConfig bcfg;
        bcfg.use_rmsnorm = config.use_rmsnorm; // C22
        blocks_.emplace_back(config.n_embd, config.n_heads, config.block_size, bcfg);
    }
}

// C23: NTK-aware base + YaRN ramp. ntk: base' = theta*scaling.
// yarn: per-dim ramp m in [0,1] between low=yarn_beta/2 and high=yarn_beta,
// freq scale = (1-m)*1/scaling + m, plus attention scale sqrt(1+0.1*ln(scaling)).
static float yarn_ramp(size_t dim, size_t n_embd, float beta) {
    float d = (float)dim / (float)n_embd * beta;
    float low = beta / 2.0f, high = beta;
    if (d < low) return 0.0f;
    if (d > high) return 1.0f;
    return (d - low) / (high - low);
}
static Tensor rope_cfg(const Tensor& x, size_t seq_len, const Config& cfg) {
    Tensor out = x;
    float scaling = cfg.rope_scaling < 1.0f ? 1.0f : cfg.rope_scaling;
    float base = cfg.rope_theta * (cfg.rope_mode == 0 ? scaling : 1.0f);
    float attn_scale = 1.0f;
    if (cfg.rope_mode == 1 && scaling > 1.0f) attn_scale = std::sqrt(1.0f + 0.1f * std::log(scaling));
    for (size_t pos = 0; pos < seq_len && pos < x.shape[0]; ++pos) {
        for (size_t i = 0; i + 1 < x.shape[1]; i += 2) {
            float freq = std::pow(base, -(float)i / x.shape[1]);
            if (cfg.rope_mode == 1 && scaling > 1.0f) {
                float m = yarn_ramp(i, x.shape[1], cfg.yarn_beta);
                freq = ((1.0f - m) / scaling + m) * std::pow(cfg.rope_theta, -(float)i / x.shape[1]);
            }
            float angle = (float)pos * freq * cfg.yarn_alpha;
            float cos_a = std::cos(angle), sin_a = std::sin(angle);
            float x0 = x(pos, i), x1 = x(pos, i + 1);
            out(pos, i) = (x0 * cos_a - x1 * sin_a) * attn_scale;
            out(pos, i + 1) = (x0 * sin_a + x1 * cos_a) * attn_scale;
        }
    }
    return out;
}
static Tensor rope(const Tensor& x, size_t seq_len) {
    Tensor out = x;
    for (size_t pos = 0; pos < seq_len && pos < x.shape[0]; ++pos) {
        for (size_t i = 0; i + 1 < x.shape[1]; i += 2) {
            float angle = pos / std::pow(10000.0f, (float)i / x.shape[1]);
            float cos_a = std::cos(angle), sin_a = std::sin(angle);
            float x0 = x(pos, i), x1 = x(pos, i + 1);
            out(pos, i) = x0 * cos_a - x1 * sin_a;
            out(pos, i + 1) = x0 * sin_a + x1 * cos_a;
        }
    }
    return out;
}
Tensor GPT::forward(const std::vector<int>& tokens) const {
    return forward_with_hidden(tokens).first;
}
std::pair<Tensor, Tensor> GPT::forward_with_hidden(const std::vector<int>& tokens) const {
    PROFILE("forward");
    size_t T = tokens.size();
    assert(T <= config_.block_size);
    Tensor x({T, config_.n_embd}, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        int tok = tokens[t];
        // Clamp OOV tokens to vocab range (modulo) to avoid UB with small test vocabs
        int tok_clamped =
            ((tok % (int)config_.vocab_size) + (int)config_.vocab_size) % (int)config_.vocab_size;
        for (size_t j = 0; j < config_.n_embd; ++j) {
            x(t, j) = wte_(tok_clamped, j) + wpe_(t, j);
        }
    }
    if (config_.pos_encoding == PosEncoding::RoPE) x = rope_cfg(x, T, config_);
    for (auto& block : blocks_) {
        x = block.forward(x);
    }
    x = config_.use_rmsnorm ? x.rmsnorm(&ln_f_gamma_) : x.layernorm(&ln_f_gamma_, &ln_f_beta_);
    Tensor logits = x.matmul(lm_head_);  // [T, vocab]
    return {logits, x};
}
void GPT::zero_grad() {
    for (auto* p : parameters()) p->zero_grad();
}
void GPT::backward(const Tensor& dlogits, const std::vector<int>& tokens, const Tensor& hidden) {
    size_t T = dlogits.shape[0];
    // grad for lm_head: hidden^T * dlogits
    Tensor dW_lm({hidden.shape[1], dlogits.shape[1]}, 0.0f);
    for (size_t i = 0; i < hidden.shape[1]; ++i)
        for (size_t j = 0; j < dlogits.shape[1]; ++j) {
            float acc = 0;
            for (size_t t = 0; t < T; ++t) acc += hidden(t, i) * dlogits(t, j);
            dW_lm(i, j) = acc;
        }
    const_cast<Tensor&>(lm_head_).add_grad(dW_lm);
    if (config_.weight_tying) {
        Tensor dWte_t({wte_.shape[0], wte_.shape[1]}, 0.0f);
        for (size_t i = 0; i < wte_.shape[0]; ++i)
            for (size_t j = 0; j < wte_.shape[1]; ++j) dWte_t(i, j) = dW_lm(j, i);
        const_cast<Tensor&>(wte_).add_grad(dWte_t);
    }
    // dHidden = dlogits * lm_head^T
    Tensor dHidden({T, hidden.shape[1]}, 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < hidden.shape[1]; ++i) {
            float acc = 0;
            for (size_t j = 0; j < dlogits.shape[1]; ++j) acc += dlogits(t, j) * lm_head_(i, j);
            dHidden(t, i) = acc;
        }
    // layernorm backward
    // Need x before final layernorm: recompute
    Tensor x({T, config_.n_embd}, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        int tok_clamped = ((tokens[t] % (int)config_.vocab_size) + (int)config_.vocab_size) %
                          (int)config_.vocab_size;
        for (size_t j = 0; j < config_.n_embd; ++j) x(t, j) = wte_(tok_clamped, j) + wpe_(t, j);
    }
    if (config_.pos_encoding == PosEncoding::RoPE) x = rope_cfg(x, T, config_);
    for (auto& blk : blocks_) x = blk.forward(x);
    // x is now pre-ln, hidden is layernorm(x)
    Tensor dX;
    if (config_.use_rmsnorm) {
        auto rb = x.rmsnorm_backward(dHidden, &ln_f_gamma_);
        dX = rb.grad_x;
        const_cast<Tensor&>(ln_f_gamma_).add_grad(rb.grad_w);
    } else {
        auto ln_bwd = x.layernorm_backward(dHidden, &ln_f_gamma_);
        dX = ln_bwd.grad_x;
        const_cast<Tensor&>(ln_f_gamma_).add_grad(ln_bwd.grad_gamma);
        const_cast<Tensor&>(ln_f_beta_).add_grad(ln_bwd.grad_beta);
    }
    // Backprop through blocks in reverse
    // Need to cache intermediates per block for accurate backward; recompute forward per block
    std::vector<Tensor> block_inputs;
    block_inputs.reserve(blocks_.size() + 1);
    Tensor cur({T, config_.n_embd}, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        int tok_clamped = ((tokens[t] % (int)config_.vocab_size) + (int)config_.vocab_size) %
                          (int)config_.vocab_size;
        for (size_t j = 0; j < config_.n_embd; ++j) cur(t, j) = wte_(tok_clamped, j) + wpe_(t, j);
    }
    if (config_.pos_encoding == PosEncoding::RoPE) cur = rope_cfg(cur, T, config_);
    block_inputs.push_back(cur);
    for (auto& blk : blocks_) {
        cur = blk.forward(cur);
        block_inputs.push_back(cur);
    }
    // Now backward
    for (int i = (int)blocks_.size() - 1; i >= 0; --i) {
        dX = blocks_[i].backward(block_inputs[i], dX);
    }
    // Grad for wte/wpe: dX is grad w.r.t. x = wte[token]+wpe[pos]
    for (size_t t = 0; t < T; ++t) {
        int tok = tokens[t];
        int tok_clamped =
            ((tok % (int)config_.vocab_size) + (int)config_.vocab_size) % (int)config_.vocab_size;
        for (size_t j = 0; j < config_.n_embd; ++j) {
            const_cast<Tensor&>(wte_).grad[tok_clamped * config_.n_embd + j] += dX(t, j);
            const_cast<Tensor&>(wpe_).grad[t * config_.n_embd + j] += dX(t, j);
        }
    }
}

std::vector<int> GPT::generate(const std::vector<int>& prompt, size_t max_new_tokens,
                               float temperature, int top_k, float top_p, float rep_penalty) const {
    std::vector<int> out = prompt;
    std::mt19937 rng(42);
    // Per-layer KV-cache: O(n) generation, each layer stores K/V for all previous tokens
    KVCache cache(config_.n_layers, config_.block_size, config_.n_embd);
    cache.clear();
    // Helper to run one incremental step for token at pos, returning logits for next token
    auto incremental_step = [&](int token, size_t pos) -> std::vector<float> {
        // Build single-token embedding
        Tensor x({1, config_.n_embd}, 0.0f);
        int tok_clamped = ((token % (int)config_.vocab_size) + (int)config_.vocab_size) % (int)config_.vocab_size;
        size_t wpe_pos = pos % config_.block_size; // wrap for sliding window
        for(size_t j=0;j<config_.n_embd;++j) x(0,j) = wte_(tok_clamped, j) + wpe_(wpe_pos, j);
        // RoPE rotation for this pos (NTK/YaRN via rope_cfg on 1×C)
        if(config_.pos_encoding == PosEncoding::RoPE){
            Tensor one({1, config_.n_embd}, 0.0f);
            for(size_t j=0;j<config_.n_embd;++j) one(0,j)=x(0,j);
            // rope_cfg expects [T,C] with T=1 at position pos: shift by building
            // a (pos+1)×C zero-padded tensor, rotating, then taking last row.
            Tensor ext({pos+1, config_.n_embd}, 0.0f);
            for(size_t j=0;j<config_.n_embd;++j) ext(pos,j)=x(0,j);
            Tensor rot = rope_cfg(ext, pos+1, config_);
            for(size_t j=0;j<config_.n_embd;++j) x(0,j)=rot(pos,j);
        }
        Tensor h = x;
        for(size_t b=0;b<blocks_.size();++b){
            h = blocks_[b].forward_incremental(h, cache, b, pos);
        }
        h = config_.use_rmsnorm ? h.rmsnorm(&ln_f_gamma_) : h.layernorm(&ln_f_gamma_, &ln_f_beta_);
        Tensor logits = h.matmul(lm_head_); // [1, vocab]
        std::vector<float> row(config_.vocab_size);
        for(size_t j=0;j<config_.vocab_size;++j) row[j]=logits(0,j);
        return row;
    };
    // Prefill prompt into cache and get initial last_logits
    std::vector<float> last_logits;
    if(!prompt.empty()){
        for(size_t pos=0;pos<prompt.size();++pos){
            last_logits = incremental_step(prompt[pos], pos);
            cache.advance(1);
        }
    } else {
        // Empty prompt: start with zero logits (uniform)
        last_logits.assign(config_.vocab_size, 0.0f);
    }
    for (size_t step = 0; step < max_new_tokens; ++step) {
        std::vector<float> cur_logits = last_logits;
        // repetition penalty
        if (rep_penalty != 1.0f)
            cur_logits = apply_repetition_penalty(cur_logits, out, rep_penalty);

        int next_id = 0;
        if (temperature == 0.0f) {
            int best = 0;
            float bestv = cur_logits[0];
            for (size_t i = 1; i < cur_logits.size(); ++i)
                if (cur_logits[i] > bestv) {
                    bestv = cur_logits[i];
                    best = (int)i;
                }
            next_id = best;
        } else {
            if (temperature != 1.0f) {
                for (auto& v : cur_logits) v /= temperature;
            }
            if (top_p < 1.0f && top_k == 0) {
                next_id = sample_top_p(cur_logits, top_p, 1.0f);
            } else if (top_k > 0) {
                std::vector<int> idx(config_.vocab_size);
                for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
                std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                                  [&](int a, int b) { return cur_logits[a] > cur_logits[b]; });
                std::uniform_int_distribution<int> dist(0, top_k - 1);
                next_id = idx[dist(rng)];
            } else {
                float maxv = cur_logits[0];
                for (size_t i = 1; i < cur_logits.size(); ++i) maxv = std::max(maxv, cur_logits[i]);
                float sum = 0;
                for (auto& v : cur_logits) {
                    v = std::exp(v - maxv);
                    sum += v;
                }
                for (auto& v : cur_logits) v /= sum;
                std::discrete_distribution<int> dist(cur_logits.begin(), cur_logits.end());
                next_id = dist(rng);
            }
        }
        out.push_back(next_id);
        if (step + 1 >= max_new_tokens) break;
        size_t pos = out.size() - 1;
        // Compute logits for next token via incremental (O(1) per step, not O(n²))
        last_logits = incremental_step(next_id, pos);
        cache.advance(1);
    }
    return out;
}

// E41: batched generate (per-sequence incremental; ragged prompts natural).
std::vector<std::vector<int>> GPT::generate_batch(const std::vector<std::vector<int>>& prompts,
                              size_t max_new_tokens, float temperature,
                              int top_k, float top_p, float rep_penalty) const {
    std::vector<std::vector<int>> outs;
    outs.reserve(prompts.size());
    for (auto& p : prompts) outs.push_back(generate(p, max_new_tokens, temperature, top_k, top_p, rep_penalty));
    return outs;
}

// E45: greedy/sampled generate recording logprob of each chosen token.
GPT::GenOutput GPT::generate_with_logprobs(const std::vector<int>& prompt, size_t max_new_tokens,
                              float temperature, int top_k, float top_p) const {
    GenOutput go;
    go.tokens = prompt;
    std::mt19937 rng(42);
    KVCache cache(config_.n_layers, config_.block_size, config_.n_embd);
    cache.clear();
    auto incremental_step = [&](int token, size_t pos) -> std::vector<float> {
        Tensor x({1, config_.n_embd}, 0.0f);
        int tc = ((token % (int)config_.vocab_size) + (int)config_.vocab_size) % (int)config_.vocab_size;
        size_t wp = pos % config_.block_size;
        for (size_t j = 0; j < config_.n_embd; ++j) x(0, j) = wte_(tc, j) + wpe_(wp, j);
        if (config_.pos_encoding == PosEncoding::RoPE) {
            Tensor ext({pos + 1, config_.n_embd}, 0.0f);
            for (size_t j = 0; j < config_.n_embd; ++j) ext(pos, j) = x(0, j);
            Tensor rot = rope_cfg(ext, pos + 1, config_);
            for (size_t j = 0; j < config_.n_embd; ++j) x(0, j) = rot(pos, j);
        }
        Tensor h = x;
        for (size_t b = 0; b < blocks_.size(); ++b) h = blocks_[b].forward_incremental(h, cache, b, pos);
        h = config_.use_rmsnorm ? h.rmsnorm(&ln_f_gamma_) : h.layernorm(&ln_f_gamma_, &ln_f_beta_);
        Tensor logits = h.matmul(lm_head_);
        std::vector<float> row(config_.vocab_size);
        for (size_t j = 0; j < config_.vocab_size; ++j) row[j] = logits(0, j);
        return row;
    };
    std::vector<float> last_logits;
    if (!prompt.empty()) {
        for (size_t pos = 0; pos < prompt.size(); ++pos) { last_logits = incremental_step(prompt[pos], pos); cache.advance(1); }
    } else last_logits.assign(config_.vocab_size, 0.0f);
    auto log_softmax_row = [](const std::vector<float>& r) {
        float m = *std::max_element(r.begin(), r.end());
        float s = 0; for (auto v : r) s += std::exp(v - m);
        float ls = m + std::log(s);
        std::vector<float> o(r.size());
        for (size_t i = 0; i < r.size(); ++i) o[i] = r[i] - ls;
        return o;
    };
    for (size_t step = 0; step < max_new_tokens; ++step) {
        auto lsm = log_softmax_row(last_logits);
        int next_id = 0;
        if (temperature == 0.0f) next_id = (int)(std::max_element(last_logits.begin(), last_logits.end()) - last_logits.begin());
        else if (top_k > 0) next_id = sample_top_k_seeded(last_logits, top_k, temperature, 42 + step);
        else if (top_p < 1.0f) next_id = sample_top_p_seeded(last_logits, top_p, temperature, 42 + step);
        else next_id = sample_temperature_seeded(last_logits, temperature, 42 + step);
        go.tokens.push_back(next_id);
        go.logprobs.push_back(lsm[next_id]);
        if (step + 1 >= max_new_tokens) break;
        size_t pos = go.tokens.size() - 1;
        last_logits = incremental_step(next_id, pos);
        cache.advance(1);
    }
    return go;
}

void GPT::save(const std::string& path) const {
    save_binary(path);
    std::cout << "[save] checkpoint written to " << path << " (" << num_parameters()
              << " params)\n";
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
    if (!out) {
        std::cerr << "[save_binary] cannot open " << path << "\n";
        return;
    }
    uint32_t magic = 0x4C4C4D00;
    out.write((char*)&magic, 4);
    uint32_t version = 3;
    out.write((char*)&version, 4);
    out.write((char*)&config_.vocab_size, sizeof(size_t));
    out.write((char*)&config_.n_layers, sizeof(size_t));
    out.write((char*)&config_.n_heads, sizeof(size_t));
    out.write((char*)&config_.n_embd, sizeof(size_t));
    out.write((char*)&config_.block_size, sizeof(size_t));
    auto write_tensor = [&](const Tensor& t) {
        uint64_t ndim = t.shape.size();
        out.write((char*)&ndim, 8);
        for (auto d : t.shape) {
            uint64_t v = d;
            out.write((char*)&v, 8);
        }
        uint64_t n = t.data.size();
        out.write((char*)&n, 8);
        out.write((char*)t.data.data(), n * sizeof(float));
    };
    write_tensor(wte_);
    write_tensor(wpe_);
    write_tensor(ln_f_gamma_);
    write_tensor(ln_f_beta_);
    write_tensor(lm_head_);
    uint64_t n_blocks = blocks_.size();
    out.write((char*)&n_blocks, 8);
    for (auto& blk : blocks_) {
        auto ps = blk.parameters();
        uint64_t n_p = ps.size();
        out.write((char*)&n_p, 8);
        for (auto* p : ps) write_tensor(*p);
    }
}
void GPT::load_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[load_binary] cannot open " << path << " (random init)\n";
        return;
    }
    uint32_t magic, version;
    in.read((char*)&magic, 4);
    in.read((char*)&version, 4);
    if (magic != 0x4C4C4D00) {
        std::cerr << "[load_binary] bad magic\n";
        return;
    }
    if (version < 1) return;
    size_t vs, nl, nh, ne, bs;
    in.read((char*)&vs, sizeof(size_t));
    in.read((char*)&nl, sizeof(size_t));
    in.read((char*)&nh, sizeof(size_t));
    in.read((char*)&ne, sizeof(size_t));
    in.read((char*)&bs, sizeof(size_t));
    bool mismatch = (vs != config_.vocab_size || nl != config_.n_layers || nh != config_.n_heads ||
                     ne != config_.n_embd || bs != config_.block_size);
    if (mismatch) {
        std::cerr << "[load_binary] config mismatch (ckpt vocab=" << vs << " n_embd=" << ne
                  << " vs current vocab=" << config_.vocab_size << " n_embd=" << config_.n_embd
                  << ") — skipping weight load, using random init\n";
        return;
    }
    auto read_tensor = [&](Tensor& t) {
        uint64_t ndim;
        in.read((char*)&ndim, 8);
        std::vector<size_t> shape(ndim);
        for (uint64_t i = 0; i < ndim; ++i) {
            uint64_t v;
            in.read((char*)&v, 8);
            shape[i] = (size_t)v;
        }
        uint64_t n;
        in.read((char*)&n, 8);
        // Use ctor to get correct strides, then read data
        t = Tensor(shape, 0.0f);
        in.read((char*)t.data.data(), n * sizeof(float));
    };
    read_tensor(wte_);
    read_tensor(wpe_);
    read_tensor(ln_f_gamma_);
    read_tensor(ln_f_beta_);
    read_tensor(lm_head_);
    if (version >= 3) {
        uint64_t n_blocks;
        in.read((char*)&n_blocks, 8);
        if (n_blocks != blocks_.size()) {
            std::cerr << "[load_binary] block count mismatch " << n_blocks << " vs "
                      << blocks_.size() << " — skipping block weights\n";
            // skip reading anyway to keep stream aligned
            for (uint64_t b = 0; b < n_blocks; ++b) {
                uint64_t n_p;
                in.read((char*)&n_p, 8);
                for (uint64_t p = 0; p < n_p; ++p) {
                    uint64_t ndim;
                    in.read((char*)&ndim, 8);
                    std::vector<size_t> shape(ndim);
                    for (uint64_t i = 0; i < ndim; ++i) {
                        uint64_t v;
                        in.read((char*)&v, 8);
                        shape[i] = v;
                    }
                    uint64_t n;
                    in.read((char*)&n, 8);
                    std::vector<float> tmp(n);
                    in.read((char*)tmp.data(), n * sizeof(float));
                }
            }
        } else {
            for (auto& blk : blocks_) {
                uint64_t n_p;
                in.read((char*)&n_p, 8);
                auto ps = blk.parameters();
                if (n_p != ps.size()) {
                    std::cerr << "[load_binary] param count mismatch\n";
                    for (uint64_t p = 0; p < n_p; ++p) {
                        uint64_t ndim;
                        in.read((char*)&ndim, 8);
                        std::vector<size_t> shape(ndim);
                        for (uint64_t i = 0; i < ndim; ++i) {
                            uint64_t v;
                            in.read((char*)&v, 8);
                            shape[i] = v;
                        }
                        uint64_t n;
                        in.read((char*)&n, 8);
                        std::vector<float> tmp(n);
                        in.read((char*)tmp.data(), n * sizeof(float));
                    }
                    continue;
                }
                for (auto* p : ps) read_tensor(*p);
            }
        }
    }
    if (config_.weight_tying) tie_weights();
}

std::vector<int> GPT::generate_streaming(const std::vector<int>& prompt, size_t max_new_tokens,
                                         std::function<void(int)> cb) const {
    // E43: true per-token streaming — cb fires as each token is sampled (not at end).
    // Uses generate_with_logprobs-equivalent incremental loop but invokes cb inline.
    std::vector<int> out = prompt;
    std::mt19937 rng(42);
    KVCache cache(config_.n_layers, config_.block_size, config_.n_embd);
    cache.clear();
    auto incremental_step = [&](int token, size_t pos) -> std::vector<float> {
        Tensor x({1, config_.n_embd}, 0.0f);
        int tc = ((token % (int)config_.vocab_size) + (int)config_.vocab_size) % (int)config_.vocab_size;
        size_t wp = pos % config_.block_size;
        for (size_t j = 0; j < config_.n_embd; ++j) x(0, j) = wte_(tc, j) + wpe_(wp, j);
        if (config_.pos_encoding == PosEncoding::RoPE) {
            Tensor ext({pos + 1, config_.n_embd}, 0.0f);
            for (size_t j = 0; j < config_.n_embd; ++j) ext(pos, j) = x(0, j);
            Tensor rot = rope_cfg(ext, pos + 1, config_);
            for (size_t j = 0; j < config_.n_embd; ++j) x(0, j) = rot(pos, j);
        }
        Tensor h = x;
        for (size_t b = 0; b < blocks_.size(); ++b) h = blocks_[b].forward_incremental(h, cache, b, pos);
        h = config_.use_rmsnorm ? h.rmsnorm(&ln_f_gamma_) : h.layernorm(&ln_f_gamma_, &ln_f_beta_);
        Tensor logits = h.matmul(lm_head_);
        std::vector<float> row(config_.vocab_size);
        for (size_t j = 0; j < config_.vocab_size; ++j) row[j] = logits(0, j);
        return row;
    };
    std::vector<float> last_logits;
    if (!prompt.empty()) {
        for (size_t pos = 0; pos < prompt.size(); ++pos) { last_logits = incremental_step(prompt[pos], pos); cache.advance(1); }
    } else last_logits.assign(config_.vocab_size, 0.0f);
    for (size_t step = 0; step < max_new_tokens; ++step) {
        int next_id = (int)(std::max_element(last_logits.begin(), last_logits.end()) - last_logits.begin());
        out.push_back(next_id);
        cb(next_id);  // stream immediately (greedy path; sampled streaming via generate_with_logprobs loop)
        if (step + 1 >= max_new_tokens) break;
        size_t pos = out.size() - 1;
        last_logits = incremental_step(next_id, pos);
        cache.advance(1);
    }
    return out;
}
void GPT::load(const std::string& path) {
    load_binary(path);
    std::cout << "[load] checkpoint loaded from " << path << "\n";
}

size_t GPT::num_parameters() const {
    size_t n =
        wte_.numel() + wpe_.numel() + ln_f_gamma_.numel() + ln_f_beta_.numel() + lm_head_.numel();
    for (auto& b : blocks_) {
        n += 4 * config_.n_embd * config_.n_embd;
        n += 4 * config_.n_embd;
        n += config_.n_embd * 4 * config_.n_embd + 4 * config_.n_embd * config_.n_embd;
        n += 5 * config_.n_embd;
    }
    if (config_.weight_tying) n -= lm_head_.numel();  // tied, not double-counted
    return n;
}

std::vector<Tensor*> GPT::parameters() {
    std::vector<Tensor*> p;
    p.reserve(4 + blocks_.size() * 14);
    p.push_back(&wte_);
    p.push_back(&wpe_);
    p.push_back(&ln_f_gamma_);
    p.push_back(&ln_f_beta_);
    if (!config_.weight_tying) p.push_back(&lm_head_);
    for (auto& blk : blocks_) {
        auto bp = blk.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    return p;
}
std::vector<const Tensor*> GPT::parameters() const {
    std::vector<const Tensor*> p;
    p.reserve(4 + blocks_.size() * 14);
    p.push_back(&wte_);
    p.push_back(&wpe_);
    p.push_back(&ln_f_gamma_);
    p.push_back(&ln_f_beta_);
    if (!config_.weight_tying) p.push_back(&lm_head_);
    for (auto& blk : blocks_) {
        auto bp = blk.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    return p;
}
void GPT::tie_weights() {
    if (!config_.weight_tying) return;
    // lm_head is [n_embd, vocab], wte is [vocab, n_embd] -> lm_head = wte^T
    if (wte_.shape.size() == 2 && lm_head_.shape.size() == 2 &&
        wte_.shape[0] == lm_head_.shape[1] && wte_.shape[1] == lm_head_.shape[0]) {
        for (size_t i = 0; i < wte_.shape[0]; ++i)
            for (size_t j = 0; j < wte_.shape[1]; ++j) lm_head_(j, i) = wte_(i, j);
    }
}

void GPT::extend_context(size_t new_block_size) {
    if (new_block_size <= config_.block_size) return;
    size_t C = config_.n_embd, old = config_.block_size;
    Tensor nw({new_block_size, C}, 0.0f);
    for (size_t p = 0; p < new_block_size; ++p) {
        // linear interp position in old table
        float src = (float)p * (float)(old - 1) / (float)(new_block_size - 1);
        size_t lo = (size_t)src, hi = std::min(lo + 1, old - 1);
        float f = src - (float)lo;
        for (size_t j = 0; j < C; ++j) nw(p, j) = wpe_(lo, j) * (1 - f) + wpe_(hi, j) * f;
    }
    wpe_ = std::move(nw);
    config_.block_size = new_block_size;
}

}  // namespace llm
