#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace llm {

class Tokenizer {
public:
    explicit Tokenizer(size_t vocab_size = 50257);

    // Simple char-level fallback (replace with BPE)
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    size_t vocab_size() const { return vocab_size_; }
    static constexpr int BOS = 0;
    static constexpr int EOS = 1;
    static constexpr int PAD = 2;
    static constexpr int UNK = 3;

    void load(const std::string& path);
    void train(const std::string& text, size_t num_merges);
    void prune_vocab(size_t keep_top_k); // keep most frequent merges
    void set_block_size(size_t new_size) { /* for context extension via RoPE scaling */ }
    void save(const std::string& path) const;
    // B12: HuggingFace compat — vocab.json {token: id} + merges.txt (#version header, "a b" lines)
    void load_hf(const std::string& vocab_path, const std::string& merges_path);
    void save_hf(const std::string& dir) const; // writes vocab.json + merges.txt
    // B13: streaming decode — holds back trailing incomplete UTF-8 sequence in `carry`
    std::string decode_incremental(const std::vector<int>& ids, std::string& carry) const;
    // B14: GPT-2 style pre-tokenizer (ASCII approx of 's|'t| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+)
    static std::vector<std::string> split_pretokenize(const std::string& text);

    // BPE merges
    std::vector<std::pair<std::string,std::string>> merges_;
private:
    size_t vocab_size_;
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> inv_vocab_;
};

} // namespace llm
