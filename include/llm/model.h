#pragma once
#include "transformer.h"
#include "tokenizer.h"
#include <vector>
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

    // Generate
    std::vector<int> generate(const std::vector<int>& prompt, size_t max_new_tokens,
                              float temperature = 1.0f, int top_k = 0) const;

    void save(const std::string& path) const;
    void save_binary(const std::string& path) const;
    void load_binary(const std::string& path);
    void load(const std::string& path);

    size_t num_parameters() const;

private:
    Config config_;
    Tensor wte_; // token embedding [vocab, n_embd]
    Tensor wpe_; // position embedding [block_size, n_embd]
    std::vector<TransformerBlock> blocks_;
    Tensor ln_f_gamma_, ln_f_beta_;
    Tensor lm_head_; // [vocab, n_embd] or [n_embd, vocab] depending on weight tying
};

} // namespace llm
