// src/mamba_scan.cpp
#include "llm/mamba_scan.h"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace llm {

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
    std::vector<float>* h_final) {
    // h[d,n] state, y[t,d] output
    // For each time step t:
    //   A_eff[d,n] = exp(A[d,n] * dt[t,d])
    //   h[t,d,n] = A_eff * h[t-1,d,n] + B[t,n] * x[t,d] * dt[t,d]
    //   y[t,d] = sum_n(C[t,n] * h[t,d,n]) + D[d] * x[t,d]
#ifdef USE_CUDA
    // Try GPU first (exact mirror); fall through to CPU on no-device/error.
    if (T > 0 && D > 0 && N > 0 && y.size() == T * D) {
        std::vector<float> y_gpu(T * D);
        if (selective_scan_cuda(x.data(), A.data(), B.data(), C.data(), dt.data(),
                                D_skip.data(), h0.data(), y_gpu.data(), T, D, N)) {
            y = std::move(y_gpu);
            if (h_final) {
                // Recompute final state on CPU (cheap O(D*N) tail is not
                // directly returned by the kernel; run reference tail).
                std::vector<float> h(D * N);
                for (size_t d = 0; d < D * N; ++d) h[d] = h0[d];
                for (size_t t = 0; t < T; ++t)
                    for (size_t d = 0; d < D; ++d) {
                        float dtv = dt[t * D + d], xv = x[t * D + d];
                        for (size_t n = 0; n < N; ++n) {
                            float a = std::exp(A[d * N + n] * dtv);
                            h[d * N + n] = a * h[d * N + n] + B[t * N + n] * xv * dtv;
                        }
                    }
                *h_final = h;
            }
            return;
        }
    }
#endif
    std::vector<float> h(D * N);
    for (size_t d = 0; d < D * N; ++d) h[d] = h0[d];

    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < D; ++d) {
            float dt_val = dt[t * D + d];
            float x_val = x[t * D + d];
            float acc = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                float a_eff = std::exp(A[d * N + n] * dt_val);
                h[d * N + n] = a_eff * h[d * N + n] + B[t * N + n] * x_val * dt_val;
                acc += C[t * N + n] * h[d * N + n];
            }
            y[t * D + d] = acc + D_skip[d] * x_val;
        }
    }
    if (h_final) *h_final = h;
}

void selective_scan_parallel(
    const std::vector<float>& x,
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C,
    const std::vector<float>& dt,
    const std::vector<float>& D_skip,
    const std::vector<float>& h0,
    std::vector<float>& y,
    size_t T, size_t D, size_t N) {
    // Chunked associative parallel scan.
    //
    // Per (d,n) the recurrence is linear: h_t = a_t*h_{t-1} + b_t with
    //   a_t = exp(A[d,n]*dt[t,d]),  b_t = B[t,n]*x[t,d]*dt[t,d].
    // Pairs compose associatively: (a2,b2) o (a1,b1) = (a2*a1, a2*b1+b2).
    // We split T into C chunks, reduce each chunk from zero to (Aprod,Bend),
    // prefix-combine chunk inits sequentially (C steps), then re-scan each
    // chunk from its true init. Chunks are independent in the second pass and
    // the outer d-loop is data-parallel (OpenMP when available). This is a
    // different code path from the single-pass sequential reference above.
    if (T == 0 || D == 0 || N == 0) return;
    if (y.size() != T * D) y.assign(T * D, 0.0f);
    // Chunk count: enough to exercise prefix logic, bounded for cache.
    size_t nChunks = std::min(T, (size_t)4);
    if (T >= 256) nChunks = std::min(T, (size_t)8);
    size_t chunk = (T + nChunks - 1) / nChunks;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long di = 0; di < (long)D; ++di) {
        size_t d = (size_t)di;
        // Per-chunk reduction from zero: Aprod[c*N+n], Bend[c*N+n].
        std::vector<float> Aprod(nChunks * N, 1.0f), Bend(nChunks * N, 0.0f);
        for (size_t c = 0; c < nChunks; ++c) {
            size_t s = c * chunk, e = std::min(T, s + chunk);
            if (s >= e) continue;
            for (size_t n = 0; n < N; ++n) {
                float p = 1.0f, z = 0.0f;
                float Adn = A[d * N + n];
                for (size_t t = s; t < e; ++t) {
                    float dtv = dt[t * D + d];
                    float a = std::exp(Adn * dtv);
                    float b = B[t * N + n] * x[t * D + d] * dtv;
                    z = a * z + b;
                    p = a * p;
                }
                Aprod[c * N + n] = p;
                Bend[c * N + n] = z;
            }
        }
        // Prefix: init[c] = state at chunk-c start given h0.
        std::vector<float> init(nChunks * N);
        for (size_t n = 0; n < N; ++n) init[n] = h0[d * N + n];
        for (size_t c = 1; c < nChunks; ++c) {
            for (size_t n = 0; n < N; ++n)
                init[c * N + n] =
                    Aprod[(c - 1) * N + n] * init[(c - 1) * N + n] + Bend[(c - 1) * N + n];
        }
        // Second pass: re-scan each chunk from its true init, emit y.
        for (size_t c = 0; c < nChunks; ++c) {
            size_t s = c * chunk, e = std::min(T, s + chunk);
            if (s >= e) continue;
            std::vector<float> h(N);
            for (size_t n = 0; n < N; ++n) h[n] = init[c * N + n];
            for (size_t t = s; t < e; ++t) {
                float dtv = dt[t * D + d];
                float xv = x[t * D + d];
                float acc = 0.0f;
                for (size_t n = 0; n < N; ++n) {
                    float a = std::exp(A[d * N + n] * dtv);
                    h[n] = a * h[n] + B[t * N + n] * xv * dtv;
                    acc += C[t * N + n] * h[n];
                }
                y[t * D + d] = acc + D_skip[d] * xv;
            }
        }
    }
}

}  // namespace llm
