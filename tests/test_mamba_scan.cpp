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

    // Chunk-boundary + randomized equivalence: the chunked associative path
    // must match the single-pass reference for uneven splits and non-trivial
    // inputs (covers T={33,129} dispatch-style boundaries and nonzero h0).
    {
        unsigned rng = 0x1234567u;
        auto rnd = [&]() {
            rng = rng * 1664525u + 1013904223u;
            return (float)(rng >> 8) / (float)(1u << 24) - 0.5f;
        };
        for (size_t TT : {33, 129}) {
            size_t DD = 16, NN = 4;
            std::vector<float> xx(TT * DD), AA(DD * NN), BB(TT * NN), CC(TT * NN),
                dd(TT * DD), Ds(DD), hh(DD * NN);
            for (auto& v : xx) v = rnd();
            for (auto& v : AA) v = -0.5f + 0.5f * rnd();
            for (auto& v : BB) v = rnd();
            for (auto& v : CC) v = rnd();
            for (auto& v : dd) v = 0.05f + 0.05f * (rnd() + 0.5f);
            for (auto& v : Ds) v = rnd();
            for (auto& v : hh) v = 0.1f * rnd();
            std::vector<float> ys(TT * DD), yp(TT * DD);
            llm::selective_scan_sequential(xx, AA, BB, CC, dd, Ds, hh, ys, TT, DD, NN);
            llm::selective_scan_parallel(xx, AA, BB, CC, dd, Ds, hh, yp, TT, DD, NN);
            float md = 0;
            for (size_t i = 0; i < TT * DD; ++i)
                md = std::max(md, std::abs(ys[i] - yp[i]));
            if (!(md < 1e-4f)) {
                std::cerr << "FAIL: T=" << TT << " par-vs-seq maxdiff=" << md << "\n";
                return 1;
            }
        }
    }

    // Explicit CUDA-kernel parity (only compiled with USE_CUDA=ON): the kernel
    // must execute on-device (no silent CPU fallback here) and match the CPU
    // reference. Under CPU builds this block does not exist.
#ifdef USE_CUDA
    {
        size_t TT = 16, DD = 32, NN = 8;
        std::vector<float> xx(TT * DD, 0.5f), AA(DD * NN, -0.7f), BB(TT * NN, 0.2f),
            CC(TT * NN, 0.9f), dd(TT * DD, 0.05f), Ds(DD, 0.5f), hh(DD * NN, 0.1f);
        std::vector<float> yg(TT * DD, 0.0f), yr(TT * DD, 0.0f);
        bool on_device = llm::selective_scan_cuda(xx.data(), AA.data(), BB.data(), CC.data(),
                                                  dd.data(), Ds.data(), hh.data(), yg.data(),
                                                  TT, DD, NN);
        if (!on_device) {
            std::cerr << "FAIL: selective_scan_cuda fell back despite CUDA build+GPU\n";
            return 1;
        }
        llm::selective_scan_parallel(xx, AA, BB, CC, dd, Ds, hh, yr, TT, DD, NN);
        float md = 0;
        for (size_t i = 0; i < TT * DD; ++i) md = std::max(md, std::abs(yg[i] - yr[i]));
        if (!(md < 1e-4f)) {
            std::cerr << "FAIL: CUDA kernel vs CPU maxdiff=" << md << "\n";
            return 1;
        }
        std::cout << "cuda kernel parity maxdiff=" << md << "\n";
    }
#endif

    std::cout << "test_mamba_scan passed\n";
    return 0;
}
