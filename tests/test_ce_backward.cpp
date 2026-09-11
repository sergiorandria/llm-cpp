#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/loss.h"
int main() {
    llm::Tensor logits({2, 4}, 0.0f);
    logits.data = {1, 0, -1, 0.5f, 0, 0, 0, 0};
    std::vector<int> tgt = {0, 2};
    // smoothing=0 must match manual softmax-1/T
    auto d = llm::cross_entropy_backward(logits, tgt, 0.0f);
    auto p = logits.softmax(1);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j) {
            float y = ((int)j == tgt[i]) ? 1.0f : 0.0f;
            assert(std::fabs(d(i, j) - (p(i, j) - y) / 2.0f) < 1e-5);
        }
    float rowsum0 = d(0, 0) + d(0, 1) + d(0, 2) + d(0, 3);
    assert(std::fabs(rowsum0) < 1e-5);  // grad rows sum to 0
    // smoothing=0.1: rows still sum to 0, target grad less negative
    auto ds = llm::cross_entropy_backward(logits, tgt, 0.1f);
    float rs = ds(0, 0) + ds(0, 1) + ds(0, 2) + ds(0, 3);
    assert(std::fabs(rs) < 1e-5);
    assert(ds(0, 0) > d(0, 0));  // smoothed target pulls less hard
    // OOV clamp path
    std::vector<int> oov = {99, -5};
    auto d2 = llm::cross_entropy_backward(logits, oov, 0.0f);
    assert(d2.shape[0] == 2);
    std::cout << "ce_backward test passed\n";
    return 0;
}
