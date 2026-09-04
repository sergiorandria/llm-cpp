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
    void save(const std::string& path) const;

    // BPE merges
    std::vector<std::pair<std::string,std::string>> merges_;
private:
    size_t vocab_size_;
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> inv_vocab_;
};

} // namespace llm
