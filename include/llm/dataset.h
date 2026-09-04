#pragma once
#include <string>
#include <vector>
namespace llm {
class Dataset {
public:
    explicit Dataset(const std::string& path, size_t block_size);
    size_t size() const { return tokens_.size(); }
    std::vector<int> get_batch(size_t idx, size_t batch_size) const;
    const std::vector<int>& tokens() const { return tokens_; }
private:
    std::vector<int> tokens_;
    size_t block_size_;
};
class DataLoader {
public:
    DataLoader(const Dataset& ds, size_t batch_size, bool shuffle=false);
    std::vector<std::vector<int>> next_batch();
    bool has_next() const;
    void reset();
private:
    const Dataset& ds_;
    size_t batch_size_;
    size_t cursor_ =0;
    bool shuffle_;
};
}
