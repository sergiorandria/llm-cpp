// tests/test_mamba_scan.cpp
#include <cassert>
#include <cmath>
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

    llm::selective_scan_sequential(x, A, B, C, dt, D_skip, h0, y, T, D, N);

    // With A=-0.5, dt=0.1: A_eff = exp(-0.5*0.1) ≈ 0.9512
    // h_0 = 0, x_0 = 1.0, B_0 = 0.1
    // h_1 = 0.9512 * 0 + 0.1 * 1.0 * 0.1 = 0.01 (B*x*dt)
    // y_0 = C_0 * h_1 + D * x_0 = 1.0 * 0.01 + 1.0 * 1.0 = 1.01
    assert(y.size() == T * D);
    assert(std::abs(y[0] - 1.01f) < 0.02f);  // rough check

    // Test that parallel scan matches sequential scan
    std::vector<float> y_parallel(T * D);
    llm::selective_scan_parallel(x, A, B, C, dt, D_skip, h0, y_parallel, T, D, N);

    for (size_t i = 0; i < T * D; ++i) {
        if (std::abs(y[i] - y_parallel[i]) > 1e-4f) {
            std::cerr << "MISMATCH at " << i << " seq=" << y[i] << " par=" << y_parallel[i] << "\n";
            assert(false);
        }
    }

    std::cout << "test_mamba_scan passed\n";
    return 0;
}
