// tests/test_mamba_block.cpp
#include <cmath>
#include <iostream>
#include "llm/mamba.h"

int main() {
    size_t n_embd = 32, d_inner = 64, d_state = 4, dt_rank = 4, conv_kernel = 4;
    llm::MambaBlock block(n_embd, d_inner, d_state, dt_rank, conv_kernel);

    // Forward pass: [T, n_embd] -> [T, n_embd]
    llm::Tensor x({4, n_embd}, 0.5f);
    llm::Tensor out = block.forward(x);
    if (out.shape[0] != 4 || out.shape[1] != n_embd) {
        std::cerr << "FAIL: output shape mismatch\n";
        return 1;
    }

    // Check no NaN
    for (size_t i = 0; i < out.data.size(); ++i) {
        if (std::isnan(out.data[i])) {
            std::cerr << "FAIL: NaN at index " << i << "\n";
            return 1;
        }
    }

    // Check residual connection: out ≈ x + small_correction
    float max_diff = 0;
    for (size_t i = 0; i < out.data.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(out.data[i] - x.data[i]));
    }
    std::cout << "max_diff from residual: " << max_diff << "\n";
    if (max_diff >= 2.0f) {
        std::cerr << "FAIL: residual difference too large (" << max_diff << ")\n";
        return 1;
    }

    // Check parameter count
    auto params = block.parameters();
    if (params.size() != 9) {
        std::cerr << "FAIL: expected 9 parameter tensors, got " << params.size() << "\n";
        return 1;
    }
    std::cout << "parameter tensors: " << params.size() << "\n";

    std::cout << "test_mamba_block passed\n";
    return 0;
}
