#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/layernorm.h"

int main() {
    // Forward sanity: zero-mean not required; scale check
    llm::RMSNorm rms(4);
    llm::Tensor x({2, 4}, 0.0f);
    x(0, 0) = 1;
    x(0, 1) = 2;
    x(0, 2) = 3;
    x(0, 3) = 4;
    x(1, 0) = -1;
    x(1, 1) = 0;
    x(1, 2) = 1;
    x(1, 3) = 2;
    auto y = rms.forward(x);
    assert(y.shape[0] == 2 && y.shape[1] == 4);
    // rms of row0 = sqrt((1+4+9+16)/4)=sqrt(7.5)
    float inv = 1.0f / std::sqrt(7.5f + 1e-5f);
    assert(std::fabs(y(0, 0) - 1 * inv) < 1e-4);

    // Backward finite-diff check on grad_x and grad_w
    llm::Tensor w({4}, 1.0f);
    w.data = {0.5f, 1.0f, 1.5f, 2.0f};
    llm::Tensor grad_out({2, 4}, 1.0f);
    auto g = x.rmsnorm_backward(grad_out, &w, 1e-5f);
    float eps = 1e-3f;
    float max_err = 0;
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j) {
            llm::Tensor xp = x, xm = x;
            xp(i, j) += eps;
            xm(i, j) -= eps;
            auto yp = xp.rmsnorm(&w), ym = xm.rmsnorm(&w);
            float num = 0;
            for (auto v : yp.data) num += v;
            float nm = 0;
            for (auto v : ym.data) nm += v;
            float fd = (num - nm) / (2 * eps);
            max_err = std::max(max_err, std::fabs(fd - g.grad_x(i, j)));
        }
    std::cout << "rmsnorm grad_x max_err=" << max_err << "\n";
    assert(max_err < 2e-2);
    // grad_w finite diff
    float max_err_w = 0;
    for (size_t j = 0; j < 4; ++j) {
        llm::Tensor wp = w, wm = w;
        wp.data[j] += eps;
        wm.data[j] -= eps;
        auto yp = x.rmsnorm(&wp), ym = x.rmsnorm(&wm);
        float sp = 0, sm = 0;
        for (auto v : yp.data) sp += v;
        for (auto v : ym.data) sm += v;
        float fd = (sp - sm) / (2 * eps);
        max_err_w = std::max(max_err_w, std::fabs(fd - g.grad_w.data[j]));
    }
    std::cout << "rmsnorm grad_w max_err=" << max_err_w << "\n";
    assert(max_err_w < 2e-2);
    std::cout << "rmsnorm test passed\n";
    return 0;
}
