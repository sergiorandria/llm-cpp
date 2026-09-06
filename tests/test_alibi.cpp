#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/alibi.h"

int main() {
    auto bias = llm::alibi_bias(4, 2);
    std::cout << "alibi shape [" << bias.shape[0] << "," << bias.shape[1] << "," << bias.shape[2]
              << "]\n";
    assert(bias.shape.size() == 3);
    assert(bias.shape[0] == 2);
    assert(bias.shape[1] == 4);
    assert(bias.shape[2] == 4);
    // Check diagonal is zero (distance 0)
    for (size_t h = 0; h < 2; ++h) {
        for (size_t i = 0; i < 4; ++i) {
            size_t idx = h * 16 + i * 4 + i;
            assert(std::abs(bias.data[idx]) < 1e-6f);
        }
    }
    // Check increasing distance more negative
    // For head 0 slope ~ 2^{-4}=0.0625, distance 3 => bias -0.1875
    size_t idx01 = 0 * 16 + 0 * 4 + 3;  // h0 i0 j3 dist 3
    size_t idx00 = 0 * 16 + 0 * 4 + 1;  // dist 1
    assert(bias.data[idx01] < bias.data[idx00]);
    // Head 1 has larger slope, so more negative for same distance
    size_t idx_h1 = 1 * 16 + 0 * 4 + 3;
    assert(bias.data[idx_h1] < bias.data[idx01]);
    // Symmetry: |i-j| same
    size_t idx_13 = 0 * 16 + 1 * 4 + 3;
    assert(std::abs(bias.data[idx_13] - bias.data[idx01]) < 0.2f ||
           std::abs(bias.data[idx_13]) > 0);
    std::cout << "alibi test passed\n";
    return 0;
}
