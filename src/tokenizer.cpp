#include "llm/tokenizer.h"
#include <fstream>

namespace llm {

Tokenizer::Tokenizer(size_t vocab_size) : vocab_size_(vocab_size) {
    // init char vocab 0-255 mapped
    for(int i=0;i<256 && i<(int)vocab_size_; ++i){
        std::string s(1, char(i));
        vocab_[s]=i;
        inv_vocab_[i]=s;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    // Placeholder: char-level encoding (0-255). Replace with BPE training.
    std::vector<int> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) {
        ids.push_back(static_cast<int>(c) % (int)vocab_size_);
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string s;
    s.reserve(ids.size());
    for (int id : ids) {
        if (id >= 0 && id < 256) s.push_back(static_cast<char>(id));
        else s.push_back('?');
    }
    return s;
}

void Tokenizer::load(const std::string& path) {
    std::ifstream in(path);
    // TODO: load BPE merges & vocab
    (void)in;
}

void Tokenizer::save(const std::string& path) const {
    std::ofstream out(path);
    (void)out;
}

} // namespace llm
