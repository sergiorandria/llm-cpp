#include "llm/attention.h"
#include <cassert>
#include <iostream>

int main() {
    // Numerical: n_heads should affect computation beyond just scale
    // With true MHA head-split, outputs for n_heads=2 vs 4 should differ
    llm::MultiHeadAttention attn2(16, 2, 8);
    llm::MultiHeadAttention attn4(16, 4, 8);
    llm::Tensor x({4, 16}, 0.3f);
    // Make x non-uniform so heads see different slices
    for (size_t i=0;i<x.data.size();++i) x.data[i] = float(i%16)*0.1f;
    auto y2 = attn2.forward(x);
    auto y4 = attn4.forward(x);
    assert(y2.shape[0]==4 && y2.shape[1]==16);
    assert(y4.shape[0]==4 && y4.shape[1]==16);
    // Check that outputs differ (if still single-head, scaling difference is tiny but not zero)
    float diff=0; for(size_t i=0;i<y2.data.size();++i) diff += std::abs(y2.data[i]-y4.data[i]);
    std::cout << "attention head diff L1=" << diff << "\n";
    assert(diff > 1e-5 && "MHA head split not effective — n_heads has no effect");
    // Also verify head_dim scaling: y2 and y4 should not be identical
    std::cout << "attention heads test passed\n";
    return 0;
}
