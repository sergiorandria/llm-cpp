// include/llm/mamba_scan.h
#pragma once
#include <vector>

namespace llm {

// Sequential selective scan: O(T*D*N) — for inference (recurrent mode).
// x: [T*D], A: [D*N], B: [T*N], C: [T*N], dt: [T*D], D_skip: [D], h0: [D*N]
// y: [T*D] output, h_final: [D*N] final state (optional, may be nullptr)
void selective_scan_sequential(
    const std::vector<float>& x,
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C,
    const std::vector<float>& dt,
    const std::vector<float>& D_skip,
    const std::vector<float>& h0,
    std::vector<float>& y,
    size_t T, size_t D, size_t N,
    std::vector<float>* h_final = nullptr);

// Parallel associative scan: O(T*log(T)*D*N) — for training.
// Same interface, but uses prefix-sum on (A,B) pairs.
void selective_scan_parallel(
    const std::vector<float>& x,
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C,
    const std::vector<float>& dt,
    const std::vector<float>& D_skip,
    const std::vector<float>& h0,
    std::vector<float>& y,
    size_t T, size_t D, size_t N);

#ifdef USE_CUDA
// CUDA entry point (src/mamba_scan_cuda.cu, linked only with USE_CUDA=ON).
// Returns false when no device / N > 64 / any CUDA error (caller: CPU fallback).
bool selective_scan_cuda(const float* x, const float* A, const float* B, const float* C,
                         const float* dt, const float* D_skip, const float* h0, float* y,
                         size_t T, size_t D, size_t N);
#endif

}  // namespace llm
