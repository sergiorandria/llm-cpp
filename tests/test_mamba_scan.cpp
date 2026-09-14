// tests/test_mamba_scan.cpp
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "llm/mamba_scan.h"

int main() {
    // Test sequential scan: h_t = A * h_{t-1} + B * x_t, y_t = C * h_t
    size_t T = 4, D = 8, N = 4;  // T=seq_len, D=d_inner, N=d_state
    std::vector<float> x(T * D, 1.0f);     // input
    std::vector<float> A(D * N, -0.5f);    // state transition (negative = decay)
    std::vector<float> B(T * N, 0.1f);     // input-dependent B
    std::vector<float> C(T * N, 1.0f);     // input-dependent C
    std::vector<float> dt(T * D, 0.1f);    // step sizes
    std::vector<float> D_skip(D, 1.0f);    // skip connection
    std::vector<float> h0(D * N, 0.0f);    // initial state
    std::vector<float> y(T * D);

    // Input size validation
    if (x.size() != T * D) { std::cerr << "FAIL: x.size() != T * D\n"; return 1; }
    if (A.size() != D * N) { std::cerr << "FAIL: A.size() != D * N\n"; return 1; }
    if (B.size() != T * N) { std::cerr << "FAIL: B.size() != T * N\n"; return 1; }
    if (C.size() != T * N) { std::cerr << "FAIL: C.size() != T * N\n"; return 1; }
    if (dt.size() != T * D) { std::cerr << "FAIL: dt.size() != T * D\n"; return 1; }
    if (D_skip.size() != D) { std::cerr << "FAIL: D_skip.size() != D\n"; return 1; }
    if (h0.size() != D * N) { std::cerr << "FAIL: h0.size() != D * N\n"; return 1; }

    llm::selective_scan_sequential(x, A, B, C, dt, D_skip, h0, y, T, D, N);

    // With A=-0.5, dt=0.1: A_eff = exp(-0.5*0.1) ≈ 0.9512
    // h_0 = 0, x_0 = 1.0, B_0 = 0.1
    // h_1 = A_eff * h_0 + B_0 * x_0 * dt = 0 + 0.1 * 1.0 * 0.1 = 0.01
    // y_0 = sum_n(C_0[n] * h_1[n]) + D * x_0 = 4 * (1.0 * 0.01) + 1.0 * 1.0 = 1.04
    if (y.size() != T * D) { std::cerr << "FAIL: y.size() != T * D\n"; return 1; }
    if (std::abs(y[0] - 1.04f) > 0.005f) {
        std::cerr << "FAIL: y[0] = " << y[0] << ", expected 1.04\n";
        return 1;
    }

    // Test that parallel scan matches sequential scan
    std::vector<float> y_parallel(T * D);
    llm::selective_scan_parallel(x, A, B, C, dt, D_skip, h0, y_parallel, T, D, N);

    for (size_t i = 0; i < T * D; ++i) {
        if (std::abs(y[i] - y_parallel[i]) > 1e-4f) {
            std::cerr << "FAIL: MISMATCH at " << i << " seq=" << y[i] << " par=" << y_parallel[i] << "\n";
            return 1;
        }
    }

    std::cout << "test_mamba_scan passed\n";
    return 0;
}
