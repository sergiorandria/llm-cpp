#include "llm/prune.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace llm {

void prune_model(GPT& model, float sparsity) {
    // Unstructured magnitude pruning: zero out smallest |w| to reach sparsity fraction
    if (sparsity <= 0) return;
    if (sparsity >= 1) {
        for (auto* p : model.parameters()) p->fill(0.0f);
        return;
    }
    for (auto* p : model.parameters()) {
        size_t n = p->data.size();
        size_t k = size_t(float(n) * sparsity);
        if (k == 0 || k >= n) continue;
        std::vector<float> abs_vals;
        abs_vals.reserve(n);
        for (float v : p->data) abs_vals.push_back(std::abs(v));
        std::nth_element(abs_vals.begin(), abs_vals.begin() + k, abs_vals.end());
        float thresh = abs_vals[k];
        for (size_t i = 0; i < n; ++i) {
            if (std::abs(p->data[i]) <= thresh) {
                // Need to ensure exactly k smallest are pruned; for ties we may overshoot
                // approximate threshold pruning
                p->data[i] = 0.0f;
            }
        }
    }
}

float sparsity(const GPT& model) {
    size_t total = 0, zeros = 0;
    for (auto* p : model.parameters()) {
        for (float v : p->data) {
            ++total;
            if (v == 0.0f) ++zeros;
        }
    }
    if (total == 0) return 0;
    return float(zeros) / float(total);
}

void prune_2to4(GPT& model) {
    for (auto* p : model.parameters()) {
        if (p->shape.size() != 2) continue;  // 2D weights only
        size_t n = p->data.size();
        for (size_t g = 0; g + 4 <= n; g += 4) {
            // find 2 smallest |.| in group
            size_t i1 = g, i2 = g + 1;
            if (std::abs(p->data[i2]) < std::abs(p->data[i1])) std::swap(i1, i2);
            for (size_t k = g + 2; k < g + 4; ++k) {
                if (std::abs(p->data[k]) < std::abs(p->data[i1])) { i2 = i1; i1 = k; }
                else if (std::abs(p->data[k]) < std::abs(p->data[i2])) { i2 = k; }
            }
            p->data[i1] = 0.0f;
            p->data[i2] = 0.0f;
        }
    }
}

bool verify_2to4(const GPT& model) {
    for (auto* p : model.parameters()) {
        if (p->shape.size() != 2) continue;
        size_t n = p->data.size();
        for (size_t g = 0; g + 4 <= n; g += 4) {
            int zeros = 0;
            for (size_t k = g; k < g + 4; ++k) if (p->data[k] == 0.0f) ++zeros;
            if (zeros != 2) return false;
        }
    }
    return true;
}

}  // namespace llm
