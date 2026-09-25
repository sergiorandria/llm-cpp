// src/mamba_scan_cuda.cu — CUDA selective scan kernel (M8).
// Built only when -DUSE_CUDA=ON with CUDAToolkit present (see CMakeLists.txt).
// One thread per d_inner channel; each thread runs the full T-step recurrence
// with its N-dim hidden state in registers (N <= 64; larger falls back to CPU).
// Returns false when no device is present or on any CUDA error so callers fall
// back to the CPU paths in mamba_scan.cpp.

#include <cuda_runtime_api.h>

namespace llm {

#define MAMBA_CUDA_MAX_N 64

__global__ void selective_scan_kernel(const float* __restrict__ x, const float* __restrict__ A,
                                      const float* __restrict__ B, const float* __restrict__ C,
                                      const float* __restrict__ dt, const float* __restrict__ Dsk,
                                      const float* __restrict__ h0, float* __restrict__ y,
                                      size_t T, size_t D, size_t N) {
    size_t d = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (d >= D) return;
    float h[MAMBA_CUDA_MAX_N];
    for (size_t n = 0; n < N; ++n) h[n] = h0[d * N + n];
    for (size_t t = 0; t < T; ++t) {
        float dtv = dt[t * D + d];
        float xv = x[t * D + d];
        float acc = 0.0f;
        for (size_t n = 0; n < N; ++n) {
            float a = expf(A[d * N + n] * dtv);
            h[n] = a * h[n] + B[t * N + n] * xv * dtv;
            acc += C[t * N + n] * h[n];
        }
        y[t * D + d] = acc + Dsk[d] * xv;
    }
}

// Host driver: exact GPU mirror of selective_scan_sequential. Returns false on
// any CUDA error / no device / N > 64 (caller falls back to CPU).
bool selective_scan_cuda(const float* x, const float* A, const float* B, const float* C,
                         const float* dt, const float* D_skip, const float* h0, float* y,
                         size_t T, size_t D, size_t N) {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) return false;
    if (T == 0 || D == 0 || N == 0) return true;
    if (N > MAMBA_CUDA_MAX_N) return false;  // CPU handles large states
    float *dx = nullptr, *dA = nullptr, *dB = nullptr, *dCc = nullptr, *ddt = nullptr,
          *dD = nullptr, *dh0 = nullptr, *dy = nullptr;
    size_t nx = T * D * sizeof(float), nA = D * N * sizeof(float), nB = T * N * sizeof(float),
           ndt = T * D * sizeof(float), nD = D * sizeof(float), nh0 = D * N * sizeof(float);
    bool ok = true;
    ok &= (cudaMalloc(&dx, nx) == cudaSuccess);
    ok &= (cudaMalloc(&dA, nA) == cudaSuccess);
    ok &= (cudaMalloc(&dB, nB) == cudaSuccess);
    ok &= (cudaMalloc(&dCc, nB) == cudaSuccess);
    ok &= (cudaMalloc(&ddt, ndt) == cudaSuccess);
    ok &= (cudaMalloc(&dD, nD) == cudaSuccess);
    ok &= (cudaMalloc(&dh0, nh0) == cudaSuccess);
    ok &= (cudaMalloc(&dy, nx) == cudaSuccess);
    if (!ok) goto cleanup;
    ok &= (cudaMemcpy(dx, x, nx, cudaMemcpyHostToDevice) == cudaSuccess);
    ok &= (cudaMemcpy(dA, A, nA, cudaMemcpyHostToDevice) == cudaSuccess);
    ok &= (cudaMemcpy(dB, B, nB, cudaMemcpyHostToDevice) == cudaSuccess);
    ok &= (cudaMemcpy(dCc, C, nB, cudaMemcpyHostToDevice) == cudaSuccess);
    ok &= (cudaMemcpy(ddt, dt, ndt, cudaMemcpyHostToDevice) == cudaSuccess);
    ok &= (cudaMemcpy(dD, D_skip, nD, cudaMemcpyHostToDevice) == cudaSuccess);
    ok &= (cudaMemcpy(dh0, h0, nh0, cudaMemcpyHostToDevice) == cudaSuccess);
    if (!ok) goto cleanup;
    {
        int threads = 256;
        int blocks = (int)((D + (size_t)threads - 1) / (size_t)threads);
        selective_scan_kernel<<<blocks, threads>>>(dx, dA, dB, dCc, ddt, dD, dh0, dy, T, D, N);
        ok &= (cudaDeviceSynchronize() == cudaSuccess);
    }
    if (ok) ok &= (cudaMemcpy(y, dy, nx, cudaMemcpyDeviceToHost) == cudaSuccess);
cleanup:
    if (dx) cudaFree(dx);
    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dCc) cudaFree(dCc);
    if (ddt) cudaFree(ddt);
    if (dD) cudaFree(dD);
    if (dh0) cudaFree(dh0);
    if (dy) cudaFree(dy);
    return ok;
}

}  // namespace llm
