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
    float max_diff = 0;
    for (size_t i = 0; i < out.data.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(out.data[i] - x.data[i]));
    }
    std::cout << "max_diff from residual: " << max_diff << "\n";
    assert(max_diff < 2.0f);

    // Check parameter count
    auto params = block.parameters();
    assert(params.size() > 5);
    std::cout << "parameter tensors: " << params.size() << "\n";

    std::cout << "test_mamba_block passed\n";
    return 0;
}
