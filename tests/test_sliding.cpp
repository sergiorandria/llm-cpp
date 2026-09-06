#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/sliding.h"

int main() {
    llm::Tensor Q({4, 4}, 0.0f), K({4, 4}, 0.0f), V({4, 4}, 0.0f);
    // Make Q/K identity-like so attention scores are predictable
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) {
            Q(i, j) = (i == j) ? 2.0f : 0.1f;
            K(i, j) = (i == j) ? 2.0f : 0.1f;
            V(i, j) = float(i + j);
        }
    // Window 2: each query should only attend to last 2 positions
    auto out = llm::sliding_attention(Q, K, V, 2);
    assert(out.shape[0] == 4 && out.shape[1] == 4);
    // Check that with window=2, the earliest token (pos0) output is finite and not NaN
    for (float v : out.data) assert(!std::isnan(v));
    // Test causal + window: for position 3, attention should be weighted towards positions 2 and 3,
    // not 0 Compute with window 4 (full causal) as reference
    auto out_full = llm::sliding_attention(Q, K, V, 4);
    // With window 2, last token's output should differ from full window because it sees fewer keys
    float diff = 0;
    for (size_t j = 0; j < 4; ++j) diff += std::abs(out(3, j) - out_full(3, j));
    std::cout << "window diff last token " << diff << "\n";
    assert(diff > 1e-4f);
    std::cout << "sliding test passed\n";
    return 0;
}
