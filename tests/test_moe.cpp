#include <cassert>
#include <iostream>

#include "llm/moe.h"

int main() {
    llm::MoEFFN moe(8, llm::MoEConfig{8, 2, 1.25f});
    llm::Tensor x({2, 8}, 0.0f);
    x.randn(0, 0.02f);
    auto out = moe.forward(x);
    assert(out.shape.size() == 2);
    assert(out.shape[0] == 2 && out.shape[1] == 8);
    for (float v : out.data) assert(!std::isnan(v) && !std::isinf(v));
    // Check sparsity: top-2 routing should produce output non-zero but limited active experts
    float norm = 0;
    for (float v : out.data) norm += v * v;
    std::cout << "moe out norm " << norm << "\n";
    assert(norm > 1e-8f);
    // Second forward with same input deterministic
    auto out2 = moe.forward(x);
    float diff = 0;
    for (size_t i = 0; i < out.data.size(); ++i) diff += std::abs(out.data[i] - out2.data[i]);
    assert(diff < 1e-6f);
    std::cout << "moe test passed\n";
    return 0;
}
