#pragma once
#include <functional>
#include <string>
#include <vector>

#include "tokenizer.h"
#include "transformer.h"

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
    std::vector<float> dropout_per_layer;
    std::vector<float> weight_decay_per_layer;
    float rope_theta = 10000.0f;  // RoPE base, NTK scaling for 8k: 50000
    float rope_scaling = 1.0f;    // 1.0 = 1k, 4.0 = 4k, 8.0 = 8k
    int rope_mode = 0;            // C23: 0 = NTK (base*scaling), 1 = YaRN ramp
    float yarn_alpha = 1.0f;      // C23: YaRN attention scale sqrt(1+0.1*ln(scaling))
    float yarn_beta = 32.0f;      // C23: ramp low/high (dims below low full scale, above high none)
    bool deterministic = false;   // A08: fixed seeds + single-thread when true
    bool use_rmsnorm = false;     // C22: RMSNorm instead of LayerNorm
    bool use_alibi = false;       // C24: ALiBi bias (mutually exclusive with RoPE)
    size_t sliding_window = 0;    // C25: 0 = full attention, else local window
    size_t global_every = 1;      // C25: every Nth layer is global when sliding
};
// C25: hybrid pattern — layer idx global iff (idx+1) % global_every == 0 (or sliding off)
inline bool is_global_layer(size_t idx, size_t global_every, size_t sliding_window) {
    if (sliding_window == 0) return true;
    if (global_every == 0) return true;
    return ((idx + 1) % global_every) == 0;
}

class GPT {
   public:
    explicit GPT(const Config& config);

    // Forward: tokens [seq_len] -> logits [seq_len, vocab_size]
    Tensor forward(const std::vector<int>& tokens) const;
    // Forward with hidden state before lm_head (for honest gradient)
    std::pair<Tensor, Tensor> forward_with_hidden(
        const std::vector<int>& tokens) const;  // {logits, hidden_ln}
    // Backward: given dlogits [T,vocab], tokens, and cached hidden, compute grads for all params
    void backward(const Tensor& dlogits, const std::vector<int>& tokens, const Tensor& hidden);
    void zero_grad();

    // Generate
    std::vector<int> generate(const std::vector<int>& prompt, size_t max_new_tokens,
                              float temperature = 1.0f, int top_k = 0, float top_p = 1.0f,
                              float rep_penalty = 1.0f) const;
    // E41: batched greedy/sampled generate — ragged prompts handled per-sequence.
    std::vector<std::vector<int>> generate_batch(const std::vector<std::vector<int>>& prompts,
                                                 size_t max_new_tokens, float temperature = 1.0f,
                                                 int top_k = 0, float top_p = 1.0f,
                                                 float rep_penalty = 1.0f) const;
    // E45: generate with per-token logprobs of the chosen token
    struct GenOutput {
        std::vector<int> tokens;
        std::vector<float> logprobs;
    };
    GenOutput generate_with_logprobs(const std::vector<int>& prompt, size_t max_new_tokens,
                                     float temperature = 1.0f, int top_k = 0,
                                     float top_p = 1.0f) const;
    // E43: true per-token streaming (cb fires as each token is sampled, greedy).
    // Matches generate(prompt, max, 0.0f) token-for-token.
    std::vector<int> generate_streaming(const std::vector<int>& prompt, size_t max_new_tokens,
                                        std::function<void(int)> cb) const;
    // legacy overload
    std::vector<int> generate_legacy(const std::vector<int>& prompt, size_t max_new_tokens,
                                     float temperature = 1.0f, int top_k = 0) const;

    void save(const std::string& path) const;
    void save_binary(const std::string& path) const;
    // H78: crash-safe write — tmp + fsync + rename, old checkpoint intact on kill
    void save_binary_atomic(const std::string& path) const;
    void load_binary(const std::string& path);
    void load(const std::string& path);

    size_t num_parameters() const;
    // Expose parameters for optimizer (non-const so optimizer can mutate)
    std::vector<Tensor*> parameters();
    std::vector<const Tensor*> parameters() const;
    // For weight tying: share storage via copy-on-tie (true alias would need shared_ptr)
    void tie_weights();
    const Config& config() const {
        return config_;
    }
    // C30: extend context by interpolating wpe_ to new_block_size (>= current)
    void extend_context(size_t new_block_size);

   private:
    Config config_;
    Tensor wte_;  // token embedding [vocab, n_embd]
    Tensor wpe_;  // position embedding [block_size, n_embd]
    std::vector<TransformerBlock> blocks_;
    Tensor ln_f_gamma_, ln_f_beta_;
    Tensor lm_head_;  // [n_embd, vocab]
};

}  // namespace llm
