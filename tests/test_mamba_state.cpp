// tests/test_mamba_state.cpp
#include <iostream>
#include "llm/mamba_state.h"

int main() {
    size_t n_layers = 2, d_inner = 16, d_state = 4;
    llm::MambaState state(n_layers, d_inner, d_state);

    if (state.size() != 0) {
        std::cerr << "FAIL: size() should be 0 initially, got " << state.size() << "\n";
        return 1;
    }
    state.clear();
    if (state.size() != 0) {
        std::cerr << "FAIL: size() should be 0 after clear(), got " << state.size() << "\n";
        return 1;
    }

    // Update layer 0
    std::vector<float> h_new(d_inner * d_state, 0.5f);
    state.update(0, h_new);
    if (state.size() != 1) {
        std::cerr << "FAIL: size() should be 1 after updating layer 0, got " << state.size() << "\n";
        return 1;
    }

    auto h = state.get(0);
    if (h.size() != d_inner * d_state) {
        std::cerr << "FAIL: get(0) size mismatch: " << h.size() << " != " << d_inner * d_state << "\n";
        return 1;
    }
    if (h[0] != 0.5f) {
        std::cerr << "FAIL: get(0)[0] = " << h[0] << ", expected 0.5\n";
        return 1;
    }

    // Update layer 1
    state.update(1, h_new);
    if (state.size() != 2) {
        std::cerr << "FAIL: size() should be 2 after updating both layers, got " << state.size() << "\n";
        return 1;
    }

    // Clear
    state.clear();
    if (state.size() != 0) {
        std::cerr << "FAIL: size() should be 0 after clear(), got " << state.size() << "\n";
        return 1;
    }

    std::cout << "test_mamba_state passed\n";
    return 0;
}
