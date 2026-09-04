#pragma once
#include "tensor.h"
#include <vector>
namespace llm {
class Embedding {
public:
    Embedding(size_t vocab, size_t dim);
    Tensor forward(const std::vector<int>& ids) const;
    size_t vocab_size() const { return vocab_; }
private:
    size_t vocab_, dim_;
    Tensor weight_;
};
}
