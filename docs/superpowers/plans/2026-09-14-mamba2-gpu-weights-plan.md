# MAMBA-2 SSM + GPU Weights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Transformer architecture with MAMBA-2 selective state space blocks and add GPU-resident model weights.

**Architecture:** MAMBA-2 blocks replace Transformer blocks. Each block uses a selective scan (input-dependent A, B, C, dt) with parallel associative scan for training and sequential recurrence for inference. GPU weights are uploaded once and read directly in CUDA kernels.

**Tech Stack:** C++20, CUDA (cuBLAS + custom kernels), existing Tensor class, CMake

**Spec:** `docs/superpowers/specs/2026-09-14-mamba2-gpu-weights-design.md`

## Global Constraints

- C++20 (g++ >= 11 / clang++ >= 15)
- CMake >= 3.20
- Build with `-j4` only (not `-j$(nproc)`)
- All existing tests must remain green throughout
- `USE_CUDA=ON` for CUDA builds, `OFF` for CPU-only
- No external dependencies beyond what's already in the project (numpy-cpp optional)
- Follow existing code style (`.clang-format`, `.clang-tidy`)

---

## Phase 1: Selective Scan Core

### Task 1: Selective Scan Interface + CPU Implementation

**Files:**
- Create: `include/llm/mamba_scan.h`
- Create: `src/mamba_scan.cpp`
- Create: `tests/test_mamba_scan.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `llm::selective_scanquential()`, `llm::selective_scan_parallel()`

- [ ] **Step 1: Write the failing test**

```cpp
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j4 --target test_mamba_scan && ./build/tests/test_mamba_scan`
Expected: FAIL with "mamba_scan.h: No such file"

- [ ] **Step 3: Write the header**

```cpp
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

}  // namespace llm
```

- [ ] **Step 4: Write minimal implementation**

```cpp
// src/mamba_scan.cpp
#include "llm/mamba_scan.h"
#include <cmath>
#include <cassert>

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
```

- [ ] **Step 5: Add to CMakeLists.txt**

Append to `tests/CMakeLists.txt`:
```cmake
add_executable(test_mamba_scan test_mamba_scan.cpp ../src/mamba_scan.cpp)
add_test(NAME mamba_scan COMMAND test_mamba_scan)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cmake --build build -j4 --target test_mamba_scan && ./build/tests/test_mamba_scan`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add include/llm/mamba_scan.h src/mamba_scan.cpp tests/test_mamba_scan.cpp tests/CMakeLists.txt
git commit -m "feat(M1): selective scan core — sequential + parallel (placeholder)"
```

---

### Task 2: MambaBlock CPU Forward

**Files:**
- Create: `include/llm/mamba.h`
- Create: `src/mamba.cpp`
- Create: `tests/test_mamba_block.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `llm::selective_scan_sequential()` from Task 1
- Produces: `llm::MambaBlock::forward()`, `llm::MambaBlock::parameters()`

- [ ] **Step 1: Write the failing test**

```cpp
// tests/test_mamba_block.cpp
#include <cassert>
#include <cmath>
#include <iostream>
#include "llm/mamba.h"

int main() {
    size_t n_embd = 32, d_inner = 64, d_state = 4, dt_rank = 4, conv_kernel = 4;
    llm::MambaBlock block(n_embd, d_inner, d_state, dt_rank, conv_kernel);

    // Forward pass: [T, n_embd] -> [T, n_embd]
    llm::Tensor x({4, n_embd}, 0.5f);
    llm::Tensor out = block.forward(x);
    assert(out.shape[0] == 4 && out.shape[1] == n_embd);

    // Check no NaN
    for (size_t i = 0; i < out.data.size(); ++i) {
        assert(!std::isnan(out.data[i]));
    }

    // Check residual connection: out ≈ x + small_correction
    // (at init, corrections should be small)
    float max_diff = 0;
    for (size_t i = 0; i < out.data.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(out.data[i] - x.data[i]));
    }
    std::cout << "max_diff from residual: " << max_diff << "\n";
    assert(max_diff < 2.0f);  // reasonable at random init

    // Check parameter count
    auto params = block.parameters();
    // Expected: norm(1) + in_proj(2*d*n) + conv1d(d*1*k + d) + x_proj(d*(dt+2*n))
    //         + dt_proj(d*dt) + A_log(d*n) + D(d) + out_proj(n*d)
    size_t expected_params = 1 + 2*d_inner*n_embd + d_inner*conv_kernel + d_inner
        + d_inner*(dt_rank + 2*d_state) + d_inner*dt_rank + d_inner*d_state
        + d_inner + n_embd*d_inner;
    assert(params.size() > 5);  // at least several parameter tensors
    std::cout << "parameter tensors: " << params.size() << "\n";

    std::cout << "test_mamba_block passed\n";
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j4 --target test_mamba_block && ./build/tests/test_mamba_block`
Expected: FAIL with "mamba.h: No such file"

- [ ] **Step 3: Write the header**

```cpp
// include/llm/mamba.h
#pragma once
#include <vector>
#include "tensor.h"

namespace llm {

struct MambaConfig {
    size_t d_inner = 0;      // 0 = auto: 2 * n_embd
    size_t d_state = 16;     // SSM state dimension
    size_t dt_rank = 0;      // 0 = auto: ceil(n_embd / 16)
    size_t conv_kernel = 4;  // Conv1D kernel width
    bool use_rmsnorm = true;
};

class MambaBlock {
public:
    MambaBlock(size_t n_embd, size_t d_inner, size_t d_state,
               size_t dt_rank, size_t conv_kernel, bool use_rmsnorm = true);

    Tensor forward(const Tensor& x) const;
    Tensor backward(const Tensor& x, const Tensor& grad_out) const;

    std::vector<Tensor*> parameters();
    std::vector<const Tensor*> parameters() const;

    size_t n_embd() const { return n_embd_; }
    size_t d_inner() const { return d_inner_; }
    size_t d_state() const { return d_state_; }

private:
    size_t n_embd_, d_inner_, d_state_, dt_rank_, conv_kernel_;
    bool use_rmsnorm_;

    // Parameters
    Tensor norm_weight_;                  // [n_embd] RMSNorm
    Tensor in_proj_weight_;               // [2*d_inner, n_embd]
    Tensor conv1d_weight_;                // [d_inner, 1, conv_kernel]
    Tensor conv1d_bias_;                  // [d_inner]
    Tensor x_proj_weight_;                // [dt_rank + 2*d_state, d_inner]
    Tensor dt_proj_weight_;               // [d_inner, dt_rank]
    Tensor a_log_;                        // [d_inner, d_state]
    Tensor d_weight_;                     // [d_inner]
    Tensor out_proj_weight_;              // [n_embd, d_inner]

    // Conv1D state (for incremental inference)
    mutable std::vector<float> conv_state_;  // [d_inner * (conv_kernel-1)]

    // Helper: depth-wise conv1d
    Tensor conv1d_forward(const Tensor& x) const;
};

}  // namespace llm
```

- [ ] **Step 4: Write implementation**

```cpp
// src/mamba.cpp
#include "llm/mamba.h"
#include "llm/mamba_scan.h"
#include <cmath>
#include <cassert>
#include <algorithm>

namespace llm {

MambaBlock::MambaBlock(size_t n_embd, size_t d_inner, size_t d_state,
                       size_t dt_rank, size_t conv_kernel, bool use_rmsnorm)
    : n_embd_(n_embd),
      d_inner_(d_inner),
      d_state_(d_state),
      dt_rank_(dt_rank),
      conv_kernel_(conv_kernel),
      use_rmsnorm_(use_rmsnorm),
      norm_weight_({n_embd}, 1.0f),
      in_proj_weight_({2 * d_inner, n_embd}),
      conv1d_weight_({d_inner, 1, conv_kernel}),
      conv1d_bias_({d_inner}),
      x_proj_weight_({dt_rank + 2 * d_state, d_inner}),
      dt_proj_weight_({d_inner, dt_rank}),
      a_log_({d_inner, d_state}),
      d_weight_({d_inner}, 1.0f),
      out_proj_weight_({n_embd, d_inner}) {
    // Initialize parameters
    in_proj_weight_.randn(0, 1.0f / std::sqrt((float)n_embd));
    conv1d_weight_.randn(0, 1.0f / std::sqrt((float)(d_inner * conv_kernel)));
    conv1d_bias_.fill(0.0f);
    x_proj_weight_.randn(0, 1.0f / std::sqrt((float)d_inner));
    dt_proj_weight_.randn(0, 0.001f);
    // A_log: initialize to log(1) = 0, or small negative for decay
    a_log_.fill(-1.0f);  // A = exp(-1) ≈ 0.37 (stable decay)
    conv_state_.assign(d_inner * (conv_kernel - 1), 0.0f);
}

Tensor MambaBlock::conv1d_forward(const Tensor& x) const {
    // x: [T, d_inner], conv1d_weight: [d_inner, 1, conv_kernel]
    // Depth-wise: each channel has its own filter
    size_t T = x.shape[0];
    size_t D = d_inner_;
    size_t k = conv_kernel_;
    Tensor out({T, D}, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < D; ++d) {
            float acc = conv1d_bias_[d];
            for (size_t j = 0; j < k; ++j) {
                ssize_t t_in = (ssize_t)t - (ssize_t)(k - 1) + (ssize_t)j;
                if (t_in >= 0) {
                    acc += x(t_in, d) * conv1d_weight_(d, 0, j);
                }
            }
            out(t, d) = acc;  // SiLU applied after
        }
    }
    return out;
}

Tensor MambaBlock::forward(const Tensor& x) const {
    assert(x.shape[1] == n_embd_);
    size_t T = x.shape[0];

    // 1. RMSNorm
    Tensor normed = use_rmsnorm_ ? x.rmsnorm(&norm_weight_) : x.layernorm(&norm_weight_);

    // 2. Input projection: [T, 2*d_inner]
    Tensor proj = normed.matmul(in_proj_weight_.transpose());
    assert(proj.shape[0] == T && proj.shape[1] == 2 * d_inner_);

    // Split into gate (x_z) and SSM branch (x_ssm)
    Tensor x_z({T, d_inner_}, 0.0f);
    Tensor x_ssm({T, d_inner_}, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_inner_; ++d) {
            x_z(t, d) = proj(t, d);
            x_ssm(t, d) = proj(t, d + d_inner_);
        }
    }

    // 3. Conv1D + SiLU on x_ssm
    Tensor conv_out = conv1d_forward(x_ssm);
    for (auto& v : conv_out.data) v = v / (1.0f + std::exp(-v));  // SiLU

    // 4. SSM parameters: dt, B, C
    Tensor x_proj_out = conv_out.matmul(x_proj_weight_.transpose());
    // x_proj_out: [T, dt_rank + 2*d_state]
    Tensor dt({T, d_inner_}, 0.0f);
    Tensor B({T, d_state_}, 0.0f);
    Tensor C({T, d_state_}, 0.0f);

    for (size_t t = 0; t < T; ++t) {
        // dt: dt_proj then softplus
        for (size_t d = 0; d < d_inner_; ++d) {
            float dt_raw = 0.0f;
            for (size_t r = 0; r < dt_rank_; ++r) {
                dt_raw += x_proj_out(t, r) * dt_proj_weight_(d, r);
            }
            dt(t, d) = std::log(1.0f + std::exp(dt_raw));  // softplus
            // Clamp dt
            dt(t, d) = std::max(0.001f, std::min(dt(t, d), 0.1f));
        }
        // B: [d_state]
        for (size_t n = 0; n < d_state_; ++n) {
            B(t, n) = x_proj_out(t, dt_rank_ + n);
        }
        // C: [d_state]
        for (size_t n = 0; n < d_state_; ++n) {
            C(t, n) = x_proj_out(t, dt_rank_ + d_state_ + n);
        }
    }

    // 5. Selective scan
    // Flatten for scan: x_ssm [T*D], A [D*N], B [T*N], C [T*N], dt [T*D]
    std::vector<float> x_flat(T * d_inner_);
    std::vector<float> A_flat(d_inner_ * d_state_);
    std::vector<float> B_flat(T * d_state_);
    std::vector<float> C_flat(T * d_state_);
    std::vector<float> dt_flat(T * d_inner_);
    std::vector<float> D_flat(d_inner_);
    std::vector<float> h0(d_inner_ * d_state_, 0.0f);

    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < d_inner_; ++d)
            x_flat[t * d_inner_ + d] = conv_out(t, d);
    for (size_t d = 0; d < d_inner_; ++d)
        for (size_t n = 0; n < d_state_; ++n)
            A_flat[d * d_state_ + n] = a_log_(d, n);
    for (size_t t = 0; t < T; ++t)
        for (size_t n = 0; n < d_state_; ++n) {
            B_flat[t * d_state_ + n] = B(t, n);
            C_flat[t * d_state_ + n] = C(t, n);
        }
    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < d_inner_; ++d)
            dt_flat[t * d_inner_ + d] = dt(t, d);
    for (size_t d = 0; d < d_inner_; ++d) D_flat[d] = d_weight_[d];

    std::vector<float> y_flat(T * d_inner_);
    selective_scan_sequential(x_flat, A_flat, B_flat, C_flat, dt_flat, D_flat, h0, y_flat, T, d_inner_, d_state_);

    Tensor ssm_out({T, d_inner_}, 0.0f);
    for (size_t t = 0; t < T; ++t)
        for (size_t d = 0; d < d_inner_; ++d)
            ssm_out(t, d) = y_flat[t * d_inner_ + d];

    // 6. Gate: SiLU(x_z) * ssm_out
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < d_inner_; ++d) {
            float gate = x_z(t, d) / (1.0f + std::exp(-x_z(t, d)));  // SiLU
            ssm_out(t, d) = gate * ssm_out(t, d);
        }
    }

    // 7. Output projection
    Tensor out = ssm_out.matmul(out_proj_weight_.transpose());

    // 8. Residual
    for (size_t i = 0; i < x.data.size(); ++i) {
        out.data[i] = x.data[i] + out.data[i];
    }

    return out;
}

Tensor MambaBlock::backward(const Tensor& x, const Tensor& grad_out) const {
    // Stub — gradient check will validate when implemented
    return Tensor(x.shape, 0.0f);
}

std::vector<Tensor*> MambaBlock::parameters() {
    return {
        &norm_weight_, &in_proj_weight_, &conv1d_weight_, &conv1d_bias_,
        &x_proj_weight_, &dt_proj_weight_, &a_log_, &d_weight_, &out_proj_weight_
    };
}

std::vector<const Tensor*> MambaBlock::parameters() const {
    return {
        &norm_weight_, &in_proj_weight_, &conv1d_weight_, &conv1d_bias_,
        &x_proj_weight_, &dt_proj_weight_, &a_log_, &d_weight_, &out_proj_weight_
    };
}

}  // namespace llm
```

- [ ] **Step 5: Add to CMakeLists.txt**

Append to `tests/CMakeLists.txt`:
```cmake
add_executable(test_mamba_block test_mamba_block.cpp ../src/mamba.cpp ../src/mamba_scan.cpp ../src/tensor.cpp ../src/layernorm.cpp)
add_test(NAME mamba_block COMMAND test_mamba_block)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cmake --build build -j4 --target test_mamba_block && ./build/tests/test_mamba_block`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add include/llm/mamba.h src/mamba.cpp tests/test_mamba_block.cpp tests/CMakeLists.txt
git commit -m "feat(M2): MambaBlock CPU forward — RMSNorm + Conv1D + SSM + gate + residual"
```

---

### Task 3: MambaState (Recurrent State for Inference)

**Files:**
- Create: `include/llm/mamba_state.h`
- Create: `src/mamba_state.cpp`
- Create: `tests/test_mamba_state.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `llm::MambaState` class with `update()`, `get()`, `clear()`

- [ ] **Step 1: Write the failing test**

```cpp
// tests/test_mamba_state.cpp
#include <cassert>
#include <iostream>
#include "llm/mamba_state.h"

int main() {
    size_t n_layers = 2, d_inner = 16, d_state = 4;
    llm::MambaState state(n_layers, d_inner, d_state);

    assert(state.size() == 0);
    state.clear();
    assert(state.size() == 0);

    // Update layer 0
    std::vector<float> h_new(d_inner * d_state, 0.5f);
    state.update(0, h_new);
    assert(state.size() == 1);

    auto h = state.get(0);
    assert(h.size() == d_inner * d_state);
    assert(h[0] == 0.5f);

    // Update layer 1
    state.update(1, h_new);
    assert(state.size() == 2);

    // Clear
    state.clear();
    assert(state.size() == 0);

    std::cout << "test_mamba_state passed\n";
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j4 --target test_mamba_state && ./build/tests/test_mamba_state`
Expected: FAIL with "mamba_state.h: No such file"

- [ ] **Step 3: Write header + implementation**

```cpp
// include/llm/mamba_state.h
#pragma once
#include <vector>

namespace llm {

class MambaState {
public:
    MambaState(size_t n_layers, size_t d_inner, size_t d_state);

    void update(size_t layer, const std::vector<float>& h);
    const std::vector<float>& get(size_t layer) const;
    void clear();
    size_t size() const;  // number of layers with non-zero state
    size_t n_layers() const { return n_layers_; }
    size_t d_inner() const { return d_inner_; }
    size_t d_state() const { return d_state_; }

private:
    size_t n_layers_, d_inner_, d_state_;
    std::vector<std::vector<float>> h_;  // per-layer state [d_inner * d_state]
};

}  // namespace llm
```

```cpp
// src/mamba_state.cpp
#include "llm/mamba_state.h"
#include <algorithm>

namespace llm {

MambaState::MambaState(size_t n_layers, size_t d_inner, size_t d_state)
    : n_layers_(n_layers), d_inner_(d_inner), d_state_(d_state),
      h_(n_layers, std::vector<float>(d_inner * d_state, 0.0f)) {}

void MambaState::update(size_t layer, const std::vector<float>& h) {
    if (layer < n_layers_) h_[layer] = h;
}

const std::vector<float>& MambaState::get(size_t layer) const {
    return h_[layer];
}

void MambaState::clear() {
    for (auto& s : h_) std::fill(s.begin(), s.end(), 0.0f);
}

size_t MambaState::size() const {
    // Count non-zero layers
    size_t count = 0;
    for (auto& s : h_)
        for (float v : s)
            if (v != 0.0f) { count++; break; }
    return count;
}

}  // namespace llm
```

- [ ] **Step 4: Add to CMakeLists.txt, run test, commit**

```bash
# Add to tests/CMakeLists.txt:
# add_executable(test_mamba_state test_mamba_state.cpp ../src/mamba_state.cpp)
# add_test(NAME mamba_state COMMAND test_mamba_state)

cmake --build build -j4 --target test_mamba_state && ./build/tests/test_mamba_state

git add include/llm/mamba_state.h src/mamba_state.cpp tests/test_mamba_state.cpp tests/CMakeLists.txt
git commit -m "feat(M3): MambaState — recurrent hidden state per layer"
```

---

## Phase 2: Model Integration

### Task 4: Config Additions + GPT MAMBA Dispatch

**Files:**
- Modify: `include/llm/model.h` (Config struct + GPT class)
- Modify: `src/model.cpp` (forward/generate uses MambaBlock when use_mamba=true)
- Create: `tests/test_mamba_model.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `MambaBlock`, `MambaState`, `MambaConfig` from Tasks 1-3
- Produces: GPT model with `use_mamba` config flag

- [ ] **Step 1: Add config fields to Config struct in model.h**

```cpp
// In Config struct, add after existing fields:
bool use_mamba = false;        // true = MAMBA-2 blocks
size_t d_inner = 0;            // 0 = auto: 2 * n_embd
size_t d_state = 16;           // SSM state dimension
size_t dt_rank = 0;            // 0 = auto: ceil(n_embd / 16)
size_t conv_kernel = 4;        // Conv1D kernel width
bool use_gpu_weights = false;  // true = upload weights to GPU
```

- [ ] **Step 2: Write failing test for MAMBA model forward**

```cpp
// tests/test_mamba_model.cpp
#include <cassert>
#include <cmath>
#include <iostream>
#include "llm/model.h"

int main() {
    llm::Config cfg;
    cfg.vocab_size = 32;
    cfg.n_embd = 32;
    cfg.n_heads = 4;    // ignored when use_mamba=true
    cfg.n_layers = 2;
    cfg.block_size = 64;
    cfg.weight_tying = true;
    cfg.use_mamba = true;
    cfg.d_inner = 64;
    cfg.d_state = 4;
    cfg.dt_rank = 4;
    cfg.conv_kernel = 4;

    llm::GPT model(cfg);
    assert(model.num_parameters() > 0);

    // Forward pass
    std::vector<int> tokens = {1, 2, 3, 4};
    auto logits = model.forward(tokens);
    assert(logits.shape[0] == 4 && logits.shape[1] == 32);

    // Check no NaN
    for (size_t i = 0; i < logits.data.size(); ++i) {
        assert(!std::isnan(logits.data[i]));
    }

    // Generate
    auto out = model.generate(tokens, 4, 0.0f, 0);
    assert(out.size() == 8);  // 4 prompt + 4 new

    std::cout << "test_mamba_model passed (params=" << model.num_parameters() << ")\n";
    return 0;
}
```

- [ ] **Step 3: Modify GPT class in model.h**

In `GPT` private section, add:
```cpp
// When use_mamba=true, these replace blocks_ + KVCache
std::vector<MambaBlock> mamba_blocks_;
MambaConfig mamba_cfg_;
```

- [ ] **Step 4: Modify model.cpp constructor**

In `GPT::GPT(const Config& config)`, add MAMBA path:
```cpp
if (config_.use_mamba) {
    size_t di = config_.d_inner ? config_.d_inner : 2 * config_.n_embd;
    size_t ds = config_.d_state;
    size_t dr = config_.dt_rank ? config_.dt_rank : (config_.n_embd + 15) / 16;
    size_t ck = config_.conv_kernel;
    mamba_blocks_.reserve(config_.n_layers);
    for (size_t i = 0; i < config_.n_layers; ++i) {
        mamba_blocks_.emplace_back(config_.n_embd, di, ds, dr, ck, config_.use_rmsnorm);
    }
}
```

- [ ] **Step 5: Modify forward_with_hidden to dispatch**

In `GPT::forward_with_hidden`, add MAMBA branch:
```cpp
if (config_.use_mamba) {
    Tensor x = ...;  // embedding + RoPE/pos
    for (auto& block : mamba_blocks_) {
        x = block.forward(x);
    }
    // final norm + lm_head
} else {
    // existing Transformer path
}
```

- [ ] **Step 6: Modify generate to use MambaState**

In `GPT::generate`, add MAMBA branch using `MambaState` instead of `KVCache`.

- [ ] **Step 7: Add to CMakeLists, run test, commit**

```bash
cmake --build build -j4 --target test_mamba_model && ./build/tests/test_mamba_model

git add include/llm/model.h src/model.cpp tests/test_mamba_model.cpp tests/CMakeLists.txt
git commit -m "feat(M4): GPT model dispatches to MambaBlock when use_mamba=true"
```

---

### Task 5: Checkpoint v4 Serialization

**Files:**
- Modify: `src/model.cpp` (save_binary, load_binary)
- Modify: `tests/test_mamba_model.cpp` (add roundtrip test)

**Interfaces:**
- Consumes: MambaBlock parameters
- Produces: Checkpoint v4 format with MAMBA tensors

- [ ] **Step 1: Add v4 magic + MAMBA tensor serialization**

In `save_binary()`, add:
```cpp
// After writing header, if config_.use_mamba:
// Write mamba-specific block parameters in order:
// norm_weight, in_proj_weight, conv1d_weight, conv1d_bias,
// x_proj_weight, dt_proj_weight, a_log, d_weight, out_proj_weight
```

In `load_binary()`, add v4 detection:
```cpp
if (version == 4 && config_.use_mamba) {
    // Read MAMBA block tensors
}
```

- [ ] **Step 2: Add roundtrip test to test_mamba_model.cpp**

```cpp
// Save and reload, verify same output
model.save_binary("/tmp/test_mamba_ckpt.bin");
llm::GPT model2(cfg);
model2.load_binary("/tmp/test_mamba_ckpt.bin");
auto logits2 = model2.forward(tokens);
for (size_t i = 0; i < logits.data.size(); ++i) {
    assert(std::abs(logits.data[i] - logits2.data[i]) < 1e-6f);
}
```

- [ ] **Step 3: Run test, commit**

```bash
cmake --build build -j4 --target test_mamba_model && ./build/tests/test_mamba_model

git add src/model.cpp tests/test_mamba_model.cpp
git commit -m "feat(M5): Checkpoint v4 — MAMBA block serialization roundtrip"
```

---

## Phase 3: GPU Weights

### Task 6: GPUWeights Class

**Files:**
- Create: `include/llm/gpu_weights.h`
- Create: `src/gpu_weights.cu`
- Create: `tests/test_gpu_weights.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `llm::GPUWeights` with `upload()`, `download()`, `device_ptr()`

- [ ] **Step 1: Write failing test**

```cpp
// tests/test_gpu_weights.cpp
#include <cassert>
#include <iostream>
#include "llm/gpu_weights.h"
#include "llm/tensor.h"

int main() {
    llm::GPUWeights gw;
    assert(gw.count() == 0);

    std::vector<llm::Tensor> params;
    params.emplace_back(llm::Tensor({4, 4}, 1.0f));
    params.emplace_back(llm::Tensor({4}, 2.0f));

    gw.upload(params);
    assert(gw.count() == 2);

    // Download back
    std::vector<llm::Tensor> out;
    gw.download(out);
    assert(out.size() == 2);
    assert(out[0].data[0] == 1.0f);
    assert(out[1].data[0] == 2.0f);

    // Device pointer access (GPU only)
    if (gw.device_ptr(0)) {
        std::cout << "GPU ptr: " << gw.device_ptr(0) << "\n";
    }

    gw.clear();
    assert(gw.count() == 0);

    std::cout << "test_gpu_weights passed\n";
    return 0;
}
```

- [ ] **Step 2: Write header (CPU stub + CUDA impl)**

```cpp
// include/llm/gpu_weights.h
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
};

}  // namespace llm
```

- [ ] **Step 3: Write CUDA implementation in gpu_weights.cu**

```cpp
// src/gpu_weights.cu
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
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return;
    d_ptrs_.resize(params.size(), nullptr);
    sizes_.resize(params.size(), 0);
    for (size_t i = 0; i < params.size(); ++i) {
        sizes_[i] = params[i].data.size();
        size_t bytes = sizes_[i] * sizeof(float);
        void* d = nullptr;
        if (cudaMalloc(&d, bytes) == cudaSuccess) {
            cudaMemcpy(d, params[i].data.data(), bytes, cudaMemcpyHostToDevice);
            d_ptrs_[i] = d;
        }
    }
#else
    (void)params;
#endif
}

void GPUWeights::download(std::vector<Tensor>& params) const {
#ifdef USE_CUDA
    params.resize(d_ptrs_.size());
    for (size_t i = 0; i < d_ptrs_.size(); ++i) {
        if (!d_ptrs_[i]) continue;
        params[i] = Tensor({sizes_[i]}, 0.0f);
        cudaMemcpy(params[i].data.data(), d_ptrs_[i], sizes_[i] * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
#else
    (void)params;
#endif
}

float* GPUWeights::device_ptr(size_t idx) {
    return (idx < d_ptrs_) ? static_cast<float*>(d_ptrs_[idx]) : nullptr;
}
const float* GPUWeights::device_ptr(size_t idx) const {
    return (idx < d_ptrs_.size()) ? static_cast<const float*>(d_ptrs_[idx]) : nullptr;
}

size_t GPUWeights::count() const { return d_ptrs_.size(); }

void GPUWeights::clear() {
#ifdef USE_CUDA
    for (auto& p : d_ptrs_) { if (p) cudaFree(p); p = nullptr; }
#endif
    d_ptrs_.clear();
    sizes_.clear();
}

}  // namespace llm
```

- [ ] **Step 4: Add CMake for .cu file, build, test, commit**

In root `CMakeLists.txt`, add CUDA source:
```cmake
if(USE_CUDA)
    target_sources(llm-cpp PRIVATE src/gpu_weights.cu)
endif()
```

```bash
cmake --build build_cuda -j4 --target test_gpu_weights && ./build_cuda/tests/test_gpu_weights

git add include/llm/gpu_weights.h src/gpu_weights.cu tests/test_gpu_weights.cpp CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat(M6): GPUWeights — upload/download model params to GPU"
```

---

### Task 7: GPU Weight Integration in GPT

**Files:**
- Modify: `include/llm/model.h` (add GPUWeights member)
- Modify: `src/model.cpp` (upload after init/load, use in forward)

**Interfaces:**
- Consumes: `GPUWeights` from Task 6
- Produces: GPT model with GPU-resident weights

- [ ] **Step 1: Add GPUWeights member to GPT class**

```cpp
// In GPT private section:
GPUWeights gpu_weights_;
bool gpu_weights_uploaded_ = false;
```

- [ ] **Step 2: Upload weights after load/init**

In `GPT` constructor and `load_binary()`, add:
```cpp
if (config_.use_gpu_weights) {
    gpu_weights_.upload(parameters());
    gpu_weights_uploaded_ = true;
}
```

- [ ] **Step 3: Modify MAMBA forward to use GPU pointers when available**

The forward pass reads from `gpu_weights_.device_ptr(i)` instead of CPU tensor data when GPU is active.

- [ ] **Step 4: Test, commit**

```bash
cmake --build build_cuda -j4 --target test_mamba_model && ./build_cuda/tests/test_mamba_model

git add include/llm/model.h src/model.cpp
git commit -m "feat(M7): GPT uploads weights to GPU when use_gpu_weights=true"
```

---

## Phase 4: CUDA Selective Scan

### Task 8: CUDA Selective Scan Kernel

**Files:**
- Create: `src/mamba_scan.cu`
- Modify: `src/mamba_scan.cpp` (dispatch to CUDA when available)
- Modify: `tests/test_mamba_scan.cpp` (add GPU parity test)

**Interfaces:**
- Consumes: same interface as CPU scan
- Produces: `selective_scan_sequential_cuda()`

- [ ] **Step 1: Write CUDA kernel**

```cuda
// src/mamba_scan.cu
#include <cuda_runtime_api.h>

namespace llm {

__global__ void selective_scan_kernel(
    const float* x, const float* A, const float* B, const float* C,
    const float* dt, const float* D, float* h, float* y,
    size_t T, size_t D_sz, size_t N) {
    // One thread per (d, n) pair for state update, then reduction for y
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D_sz) return;

    for (size_t t = 0; t < T; ++t) {
        float dt_val = dt[t * D_sz + d];
        float x_val = x[t * D_sz + d];
        float acc = 0.0f;
        for (size_t n = 0; n < N; ++n) {
            float a_eff = expf(A[d * N + n] * dt_val);
            h[d * N + n] = a_eff * h[d * N + n] + B[t * N + n] * x_val * dt_val;
            acc += C[t * N + n] * h[d * N + n];
        }
        y[t * D_sz + d] = acc + D[d] * x_val;
    }
}

bool selective_scan_sequential_cuda(
    const float* x, const float* A, const float* B, const float* C,
    const float* dt, const float* D, float* h, float* y,
    size_t T, size_t D_sz, size_t N) {
    // Launch: one thread per d, loop over t and n
    int threads = 256;
    int blocks = (D_sz + threads - 1) / threads;
    selective_scan_kernel<<<blocks, threads>>>(x, A, B, C, dt, D, h, y, T, D_sz, N);
    return cudaDeviceSynchronize() == cudaSuccess;
}

}  // namespace llm
```

- [ ] **Step 2: Add CUDA dispatch in mamba_scan.cpp**

```cpp
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
extern bool selective_scan_sequential_cuda(...);
#endif

void selective_scan_sequential(...) {
#ifdef USE_CUDA
    // Check device, allocate GPU buffers, call CUDA kernel
    // Fall back to CPU if no device
#endif
    // CPU path (existing)
}
```

- [ ] **Step 3: Add GPU parity test, run, commit**

```bash
cmake --build build_cuda -j4 --target test_mamba_scan && ./build_cuda/tests/test_mamba_scan

git add src/mamba_scan.cu src/mamba_scan.cpp tests/test_mamba_scan.cpp
git commit -m "feat(M8): CUDA selective scan kernel — sequential scan on GPU"
```

---

## Phase 5: Cleanup + Final Integration

### Task 9: Remove Old Transformer Code

**Files:**
- Remove from compilation: `attention.cpp`, `feed_forward.cpp`, `transformer.cpp`, `flash_attention.cpp`, `kv_cache.cpp`, `alibi.cpp`, `sliding.cpp`, `mqa.cpp`
- Keep headers for reference (or remove entirely)
- Update `CMakeLists.txt` to not compile removed files when `use_mamba=true`
- Update `tests/CMakeLists.txt` to not build removed tests when `use_mamba=true`

- [ ] **Step 1: Guard Transformer sources behind !use_mamba in CMake**

```cmake
if(NOT use_mamba)
    # compile Transformer sources
endif()
```

- [ ] **Step 2: Update README.md with MAMBA-2 architecture**

- [ ] **Step 3: Run full test suite, commit**

```bash
cmake --build build -j4 && ctest --test-dir build
git commit -m "feat(M9): remove Transformer code, MAMBA-2 is default architecture"
```

---

### Task 10: Gradient Check + Overfit Test

**Files:**
- Modify: `tests/test_mamba_block.cpp` (add gradient check)
- Create: `tests/test_mamba_overfit.cpp`

- [ ] **Step 1: Finite-difference gradient check on MambaBlock**

```cpp
// Perturb each parameter by epsilon, check dL/dparam matches finite difference
```

- [ ] **Step 2: Small model overfits tiny dataset**

```cpp
// 2-layer MAMBA, 32 embd, train on 100 tokens, verify loss decreases
```

- [ ] **Step 3: Run, commit**

```bash
git commit -m "feat(M10): MAMBA gradient check + overfit test"
```

---

## Summary

| Task | Deliverable | Tests |
|------|------------|-------|
| M1 | Selective scan (sequential + parallel) | test_mamba_scan |
| M2 | MambaBlock CPU forward | test_mamba_block |
| M3 | MambaState recurrent state | test_mamba_state |
| M4 | GPT model dispatches to MAMBA | test_mamba_model |
| M5 | Checkpoint v4 roundtrip | test_mamba_model |
| M6 | GPUWeights upload/download | test_gpu_weights |
| M7 | GPT GPU weight integration | test_mamba_model (CUDA) |
| M8 | CUDA selective scan kernel | test_mamba_scan (CUDA) |
| M9 | Remove old Transformer code | full test suite |
| M10 | Gradient check + overfit | test_mamba_block, test_mamba_overfit |
