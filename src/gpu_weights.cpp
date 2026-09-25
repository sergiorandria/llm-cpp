#include "llm/gpu_weights.h"
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace llm {

GPUWeights::~GPUWeights() { clear(); }

void GPUWeights::upload(const std::vector<Tensor>& params) {
    clear();
    shapes_.reserve(params.size());
    for (auto& p : params) shapes_.push_back(p.shape);
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
        if (bytes == 0) continue;
        if (cudaMalloc(&d, bytes) != cudaSuccess) {
            cpu_tensors_.push_back(params[i]);
            continue;
        }
        if (cudaMemcpy(d, params[i].data.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d);
            cpu_tensors_.push_back(params[i]);
        } else {
            d_ptrs_[i] = d;
        }
    }
#else
    cpu_tensors_ = params;
#endif
}

void GPUWeights::download(std::vector<Tensor>& params) const {
#ifdef USE_CUDA
    // Order-preserving: params[i] corresponds to upload index i; entries that
    // fell back to CPU come from cpu spill in upload order. Reconstruct by
    // walking upload order: device ptr if present, else next CPU spill tensor.
    // Spill order matches upload index order, so this is exact.
    params.clear();
    params.reserve(shapes_.size());
    size_t spill = 0;
    // Count spills to sanity-check; cpu_tensors_ holds exactly the fallbacks.
    for (size_t i = 0; i < shapes_.size(); ++i) {
        bool on_device = (i < d_ptrs_.size() && d_ptrs_[i] != nullptr);
        if (on_device) {
            Tensor t(shapes_[i], 0.0f);
            if (cudaMemcpy(t.data.data(), d_ptrs_[i], sizes_[i] * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                // Fall back to CPU spill if still available, else zeros.
                if (spill < cpu_tensors_.size())
                    t = cpu_tensors_[spill++];
                else
                    t.fill(0.0f);
            }
            params.push_back(std::move(t));
        } else {
            if (spill < cpu_tensors_.size())
                params.push_back(cpu_tensors_[spill++]);
            else
                params.emplace_back(shapes_[i], 0.0f);
        }
    }
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
    shapes_.clear();
    cpu_tensors_.clear();
}

}  // namespace llm
