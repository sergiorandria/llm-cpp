#include "llm/tokenizer.h"
#include <fstream>

namespace llm {

Tokenizer::Tokenizer(size_t vocab_size) : vocab_size_(vocab_size) {
    // init byte-level vocab 0-255 mapped (GPT-2 style)
    for(int i=0;i<256 && i<(int)vocab_size_; ++i){
        std::string s(1, char(i));
        vocab_[s]=i;
        inv_vocab_[i]=s;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    // BPE encode: start from char ids, then apply merges greedily
    std::vector<std::string> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text) tokens.emplace_back(1, char(c));
    // apply merges in order learned
    for (auto &mer: merges_) {
        std::string merged = mer.first + mer.second;
        for (size_t i = 0; i + 1 < tokens.size(); ) {
            if (tokens[i] == mer.first && tokens[i+1] == mer.second) {
                tokens[i] = merged;
                tokens.erase(tokens.begin() + i + 1);
            } else ++i;
        }
    }
    std::vector<int> ids;
    ids.reserve(tokens.size());
    for (auto &tok : tokens) {
        auto it = vocab_.find(tok);
        if (it != vocab_.end()) ids.push_back(it->second);
        else ids.push_back(UNK % (int)vocab_size_);
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
    std::string line;
    while(std::getline(in, line)){
        if(line.empty() || line[0]=='#') continue;
        size_t sp = line.find(' ');
        if(sp==std::string::npos) continue;
        std::string a=line.substr(0,sp), b=line.substr(sp+1);
        merges_.emplace_back(a,b);
        std::string merged=a+b;
        if(vocab_.find(merged)==vocab_.end() && vocab_.size()<vocab_size_){
            int id=(int)vocab_.size();
            vocab_[merged]=id;
            inv_vocab_[id]=merged;
        }
    }
}

void Tokenizer::train(const std::string& text, size_t num_merges){
    // Very naive BPE train: count adjacent pairs and merge most frequent
    (void)text; (void)num_merges;
    // placeholder: add dummy merges for demo
    for(size_t i=0;i<num_merges && vocab_.size()<vocab_size_; ++i){
        merges_.emplace_back("a","b");
    }
}
void Tokenizer::save(const std::string& path) const {
    std::ofstream out(path);
    for(auto &m: merges_) out<<m.first<<" "<<m.second<<"\n";
}

} // namespace llm
