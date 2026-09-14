#include "llm/gpu_weights.h"
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace llm {

GPUWeights::~GPUWeights() { clear(); }

void GPUWeights::upload(const std::vector<Tensor>& params) {
    clear();
#ifdef USE_CUDA
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        cpu_tensors_ = params;
        return;
    }
    d_ptrs_.resize(params.size(), nullptr);
    sizes_.resize(params.size(), 0);
    for (size_t i = 0; i < params.size(); ++i) {
        sizes_[i] = params[i].data.size();
        size_t bytes = sizes_[i] * sizeof(float);
        void* d = nullptr;
        if (cudaMalloc(&d, bytes) == cudaSuccess) {
            cudaMemcpy(d, params[i].data.data(), bytes, cudaMemcpyHostToDevice);
            d_ptrs_[i] = d;
        } else {
            cpu_tensors_.push_back(params[i]);
        }
    }
#else
    cpu_tensors_ = params;
#endif
}

void GPUWeights::download(std::vector<Tensor>& params) const {
#ifdef USE_CUDA
    params.resize(d_ptrs_.size() + cpu_tensors_.size());
    size_t idx = 0;
    for (size_t i = 0; i < d_ptrs_.size(); ++i) {
        if (!d_ptrs_[i]) continue;
        params[idx] = Tensor({sizes_[i]}, 0.0f);
        cudaMemcpy(params[idx].data.data(), d_ptrs_[i], sizes_[i] * sizeof(float),
                   cudaMemcpyDeviceToHost);
        ++idx;
    }
    for (size_t i = 0; i < cpu_tensors_.size(); ++i) {
        params[idx++] = cpu_tensors_[i];
    }
    params.resize(idx);
#else
    params = cpu_tensors_;
#endif
}

float* GPUWeights::device_ptr(size_t idx) {
#ifdef USE_CUDA
    return (idx < d_ptrs_.size()) ? static_cast<float*>(d_ptrs_[idx]) : nullptr;
#else
    (void)idx;
    return nullptr;
#endif
}
const float* GPUWeights::device_ptr(size_t idx) const {
#ifdef USE_CUDA
    return (idx < d_ptrs_.size()) ? static_cast<const float*>(d_ptrs_[idx]) : nullptr;
#else
    (void)idx;
    return nullptr;
#endif
}

size_t GPUWeights::count() const {
#ifdef USE_CUDA
    return d_ptrs_.size() + cpu_tensors_.size();
#else
    return cpu_tensors_.size();
#endif
}

void GPUWeights::clear() {
#ifdef USE_CUDA
    for (auto& p : d_ptrs_) {
        if (p) cudaFree(p);
        p = nullptr;
    }
#endif
    d_ptrs_.clear();
    sizes_.clear();
    cpu_tensors_.clear();
}

}  // namespace llm
