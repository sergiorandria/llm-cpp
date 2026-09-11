#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/optimizer.h"
int main() {
    // 1D params excluded from decay: with zero grad, 1D must stay, 2D must shrink by lr*wd
    llm::Tensor w2d({2, 2}, 1.0f);
    llm::Tensor b1d({2}, 1.0f);
    std::vector<llm::Tensor*> params = {&w2d, &b1d};
    llm::Tensor g2({2, 2}, 0.0f), g1({2}, 0.0f);
    std::vector<llm::Tensor> grads = {g2, g1};
    llm::AdamW opt(0.1f, 0.9f, 0.999f, 1e-8f, 0.01f);
    opt.step(params, grads);
    // w decayed: 1 - 0.1*0.01 = 0.999 (Adam step with 0 grad is no-op)
    assert(std::fabs(w2d.data[0] - 0.999f) < 1e-5);
    assert(std::fabs(b1d.data[0] - 1.0f) < 1e-6);
    // Explicit flags override: force decay on 1D
    llm::Tensor b2({2}, 1.0f);
    std::vector<llm::Tensor*> p2 = {&b2};
    std::vector<llm::Tensor> g02 = {g1};
    opt.step(p2, g02, std::vector<char>{1});
    assert(std::fabs(b2.data[0] - 0.999f) < 1e-5);
    // Rosenbrock decreases over 20 Adam steps (bias correction sanity)
    llm::Tensor p({2}, 0.0f);
    p.data = {-1.0f, 1.0f};
    llm::Adam adam(0.01f);
    float prev = 1e30f;
    for (int t = 0; t < 20; ++t) {
        float x = p.data[0], y = p.data[1];
        float loss = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
        (void)loss;
        llm::Tensor g({2}, 0.0f);
        g.data[0] = -2 * (1 - x) - 400 * x * (y - x * x);
        g.data[1] = 200 * (y - x * x);
        std::vector<llm::Tensor*> pp = {&p};
        std::vector<llm::Tensor> gg = {g};
        adam.step(pp, gg);
        float nl = (1 - p.data[0]) * (1 - p.data[0]) +
                   100 * (p.data[1] - p.data[0] * p.data[0]) * (p.data[1] - p.data[0] * p.data[0]);
        if (t > 5) assert(nl < prev + 1e-3);
        prev = nl;
    }
    std::cout << "adamw test passed\n";
    return 0;
}
