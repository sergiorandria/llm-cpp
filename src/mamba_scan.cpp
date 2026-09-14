// src/mamba_scan.cpp
#include "llm/mamba_scan.h"
#include <cmath>

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
    // Simplified: for now, parallel scan = sequential scan (correctness first).
    // Real parallel scan uses associative prefix-sum on (A,B) pairs.
    // Will optimize in a later task.
    std::vector<float> h_final(D * N);
    selective_scan_sequential(x, A, B, C, dt, D_skip, h0, y, T, D, N, &h_final);
}

}  // namespace llm
