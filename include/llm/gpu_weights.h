#pragma once
#include <vector>
#include "tensor.h"

namespace llm {

class GPUWeights {
public:
    GPUWeights() = default;
    ~GPUWeights();

    void upload(const std::vector<Tensor>& params);
    void download(std::vector<Tensor>& params) const;
    float* device_ptr(size_t idx);
    const float* device_ptr(size_t idx) const;
    size_t count() const;
    void clear();

 private:
    std::vector<void*> d_ptrs_;
    std::vector<size_t> sizes_;
    std::vector<std::vector<size_t>> shapes_;  // per-param shapes (download restores them)
    std::vector<Tensor> cpu_tensors_;
};

}  // namespace llm
