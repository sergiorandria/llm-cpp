#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/optimizer.h"
int main() {
    llm::Tensor g1({2}, 0.0f);
    g1.data = {3.0f, 4.0f};  // norm 5
    llm::Tensor g2({1}, 0.0f);
    g2.data = {0.0f};
    std::vector<llm::Tensor> grads = {g1, g2};
    float before = llm::clip_by_global_norm(grads, 1.0f);
    assert(std::fabs(before - 5.0f) < 1e-4);
    float n = 0;
    for (auto& g : grads)
        for (auto v : g.data) n += v * v;
    assert(std::fabs(std::sqrt(n) - 1.0f) < 1e-4);
    // small norm untouched
    std::vector<llm::Tensor> g3 = {llm::Tensor({2}, 0.0f)};
    g3[0].data = {0.3f, 0.4f};  // norm 0.5
    float b2 = llm::clip_by_global_norm(g3, 1.0f);
    assert(std::fabs(b2 - 0.5f) < 1e-4);
    assert(std::fabs(g3[0].data[0] - 0.3f) < 1e-6);
    std::cout << "clip test passed\n";
    return 0;
}
