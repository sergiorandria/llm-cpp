#include "llm/dataset.h"

#include <algorithm>
#include <fstream>
#include <random>

#include "llm/tokenizer.h"
#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace llm {
// B15: read file bytes — mmap path avoids an extra user-space copy of the raw text
static std::string read_file_bytes(const std::string& path, bool use_mmap) {
#ifdef __linux__
    if (use_mmap) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            struct stat st;
            if (fstat(fd, &st) == 0 && st.st_size > 0) {
                void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
                if (m != MAP_FAILED) {
                    std::string s((const char*)m, st.st_size);
                    munmap(m, st.st_size);
                    close(fd);
                    return s;
                }
            }
            close(fd);
        }
        // fall through to ifstream on any failure
    }
#else
    (void)use_mmap;
#endif
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), {});
}
Dataset::Dataset(const std::string& path, size_t block_size, bool use_mmap)
    : block_size_(block_size) {
    std::string text = read_file_bytes(path, use_mmap);
    if (text.empty()) {
        // try generate tiny shakespeare stub, still use tokenizer
        text = "To be, or not to be, that is the question.\n";
        for (int i = 0; i < 10; ++i) text += text;
    }
    Tokenizer tok;
    tokens_ = tok.encode(text);
    if (tokens_.empty()) tokens_.assign(256, 0);
}
std::vector<int> Dataset::get_batch(size_t idx, size_t batch_size) const {
    std::vector<int> batch;
    for (size_t i = 0; i < batch_size && idx + i < tokens_.size(); ++i)
        batch.push_back(tokens_[idx + i]);
    return batch;
}
DataLoader::DataLoader(const Dataset& ds, size_t batch_size, bool shuffle, uint64_t seed)
    : ds_(ds), batch_size_(batch_size), shuffle_(shuffle), seed_(seed) {
    build_order();
}
void DataLoader::build_order() {
    order_.clear();
    for (size_t s = 0; s < ds_.size(); s += batch_size_) order_.push_back(s);
    if (shuffle_) {
        std::mt19937_64 rng(seed_);
        std::shuffle(order_.begin(), order_.end(), rng);
    }
}
std::vector<std::vector<int>> DataLoader::next_batch() {
    std::vector<std::vector<int>> b;
    if (cursor_ >= order_.size()) return b;
    b.push_back(ds_.get_batch(order_[cursor_], batch_size_));
    cursor_++;
    return b;
}
bool DataLoader::has_next() const {
    return cursor_ < order_.size();
}
void DataLoader::reset() {
    cursor_ = 0;
    if (shuffle_) build_order();
}

// ── B16 ──
PackedBatch pack_with_eos(const std::vector<int>& tokens, size_t block_size, int eos_id) {
    PackedBatch out;
    std::vector<int> cur;
    auto flush = [&]() {
        std::vector<int> row(block_size, eos_id), m(block_size, 0);
        size_t n = std::min(cur.size(), block_size);
        for (size_t i = 0; i < n; ++i) {
            row[i] = cur[i];
            m[i] = 1;
        }
        out.input.push_back(std::move(row));
        out.mask.push_back(std::move(m));
        cur.clear();
    };
    for (int t : tokens) {
        cur.push_back(t);
        if (cur.size() == block_size) flush();
    }
    if (!cur.empty()) flush();
    return out;
}

// ── B18 ──
void train_val_split(const std::vector<int>& tokens, double train_ratio, uint64_t seed,
                     std::vector<int>& train_out, std::vector<int>& val_out) {
    // Deterministic contiguous split (seed reserved for future shuffling; kept for API stability).
    (void)seed;
    size_t n_train = (size_t)(tokens.size() * train_ratio);
    train_out.assign(tokens.begin(), tokens.begin() + n_train);
    val_out.assign(tokens.begin() + n_train, tokens.end());
}
}  // namespace llm
