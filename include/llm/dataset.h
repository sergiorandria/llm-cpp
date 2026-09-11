#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace llm {
class Dataset {
   public:
    explicit Dataset(const std::string& path, size_t block_size, bool use_mmap = false);
    size_t size() const {
        return tokens_.size();
    }
    std::vector<int> get_batch(size_t idx, size_t batch_size) const;
    const std::vector<int>& tokens() const {
        return tokens_;
    }

   private:
    std::vector<int> tokens_;
    size_t block_size_;
};
// B16: packing with EOS separators + attention masks (1 = real, 0 = pad)
struct PackedBatch {
    std::vector<std::vector<int>> input;  // [B, block_size]
    std::vector<std::vector<int>> mask;
};
PackedBatch pack_with_eos(const std::vector<int>& tokens, size_t block_size, int eos_id);
// B18: deterministic train/val split by token ratio
void train_val_split(const std::vector<int>& tokens, double train_ratio, uint64_t seed,
                     std::vector<int>& train_out, std::vector<int>& val_out);
class DataLoader {
   public:
    // B17: shuffle uses Fisher-Yates with seed (deterministic); seed ignored if !shuffle
    DataLoader(const Dataset& ds, size_t batch_size, bool shuffle = false, uint64_t seed = 42);
    std::vector<std::vector<int>> next_batch();
    bool has_next() const;
    void reset();

   private:
    const Dataset& ds_;
    size_t batch_size_;
    size_t cursor_ = 0;
    bool shuffle_;
    uint64_t seed_;
    std::vector<size_t> order_;  // block start indices (shuffled when shuffle_)
    void build_order();
};
}  // namespace llm
