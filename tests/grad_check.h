#pragma once
// Central finite-difference grad-check helper (PRODUCTION_100 A02).
#include <cmath>
#include <functional>

#include "llm/tensor.h"

namespace llm::test {

// Sum all elements (scalar loss proxy for grad-check).
inline float sum_all(const Tensor& t) {
    float s = 0;
    for (auto v : t.data) s += v;
    return s;
}

// Max abs error between analytic grad and central finite differences.
// fn maps perturbed input -> output tensor; loss = sum(output).
inline float max_fd_error(Tensor x, std::function<Tensor(const Tensor&)> fn,
                          const Tensor& analytic_grad, float eps = 1e-3f) {
    float max_err = 0;
    for (size_t i = 0; i < x.data.size(); ++i) {
        Tensor xp = x, xm = x;
        xp.data[i] += eps;
        xm.data[i] -= eps;
        float lp = sum_all(fn(xp));
        float lm = sum_all(fn(xm));
        float fd = (lp - lm) / (2 * eps);
        max_err = std::max(max_err, std::fabs(fd - analytic_grad.data[i]));
    }
    return max_err;
}

}  // namespace llm::test
