#pragma once
#include "transformer.h"
#include "tokenizer.h"
#include <vector>
#include <functional>
#include <string>

namespace llm {

enum class PosEncoding { Learned, Sinusoidal, RoPE };

struct Config {
    size_t vocab_size = 50257;
    size_t n_layers = 12;
    size_t n_heads = 12;
    size_t n_embd = 768;
    size_t block_size = 1024;
    float dropout = 0.0f;
    bool bias = false;
    PosEncoding pos_encoding = PosEncoding::Learned;
    bool weight_tying = true;
};

class GPT {
public:
    explicit GPT(const Config& config);

    // Forward: tokens [seq_len] -> logits [seq_len, vocab_size]
    Tensor forward(const std::vector<int>& tokens) const;
    // Forward with hidden state before lm_head (for honest gradient)
    std::pair<Tensor, Tensor> forward_with_hidden(const std::vector<int>& tokens) const; // {logits, hidden_ln}

    // Generate
    std::vector<int> generate(const std::vector<int>& prompt, size_t max_new_tokens,
                              float temperature = 1.0f, int top_k = 0, float top_p = 1.0f, float rep_penalty=1.0f) const;
    std::vector<int> generate_streaming(const std::vector<int>& prompt, size_t max_new_tokens, std::function<void(int)> cb) const;
    // legacy overload
    std::vector<int> generate_legacy(const std::vector<int>& prompt, size_t max_new_tokens,
                              float temperature = 1.0f, int top_k = 0) const;

    void save(const std::string& path) const;
    void save_binary(const std::string& path) const;
    void load_binary(const std::string& path);
    void load(const std::string& path);

    size_t num_parameters() const;
    // Expose parameters for optimizer (non-const so optimizer can mutate)
    std::vector<Tensor*> parameters();
    std::vector<const Tensor*> parameters() const;
    // For weight tying: share storage via copy-on-tie (true alias would need shared_ptr)
    void tie_weights();

private:
    Config config_;
    Tensor wte_; // token embedding [vocab, n_embd]
    Tensor wpe_; // position embedding [block_size, n_embd]
    std::vector<TransformerBlock> blocks_;
    Tensor ln_f_gamma_, ln_f_beta_;
    Tensor lm_head_; // [n_embd, vocab]
};

} // namespace llm
