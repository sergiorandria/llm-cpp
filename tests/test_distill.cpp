#include <cassert>
#include <iostream>

#include "llm/distill.h"

int main() {
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_embd = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 1;
    cfg.block_size = 8;
    llm::GPT teacher(cfg);
    llm::GPT student(cfg);
    // Ensure teacher != student (otherwise KL grad is zero)
    for (auto* p : teacher.parameters()) {
        for (auto& v : p->data) v += 0.1f;
        break;
    }
    std::vector<int> batch = {1, 2, 3, 4};
    // Capture student params before
    auto params_before = student.parameters();
    std::vector<float> before;
    for (auto* p : params_before) before.insert(before.end(), p->data.begin(), p->data.end());
    llm::distill_step(student, teacher, batch, 2.0f);
    // Check grads were accumulated (need at least one non-zero grad)
    bool has_grad = false;
    for (auto* p : student.parameters()) {
        for (float g : p->grad)
            if (std::abs(g) > 1e-8f) {
                has_grad = true;
                break;
            }
    }
    std::cout << "distill has_grad " << has_grad << "\n";
    if (!has_grad) {
        std::cerr << "distill has_grad failed\n";
        return 1;
    }
    assert(has_grad);
    // Do an optimizer step to ensure update works
    float sum_before = 0;
    for (float v : before) sum_before += v;
    // simple sgd with larger lr to make diff observable even for small grads
    for (auto* p : student.parameters())
        for (size_t i = 0; i < p->data.size(); ++i) p->data[i] -= 1e-1f * p->grad[i];
    auto params_after = student.parameters();
    float diff = 0;
    size_t idx = 0;
    for (auto* p : params_after)
        for (float v : p->data) diff += std::abs(v - before[idx++]);
    std::cout << "distill param diff " << diff << "\n";
    if (!(diff > 1e-9f)) {
        std::cerr << "distill param diff failed (diff=" << diff << ")\n";
        return 1;
    }
    assert(diff > 1e-9f);
    std::cout << "distill test passed\n";
    return 0;
}
