#pragma once
#include "tensor.h"
#include <vector>
namespace llm {
Tensor sinusoidal_pos_embedding(size_t seq_len, size_t dim);
class PosEmbedding {
public:
    PosEmbedding(size_t max_len, size_t dim, bool learned=true);
    Tensor forward(size_t seq_len) const;
private:
    Tensor weight_;
    bool learned_;
};
}
