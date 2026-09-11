#include <cassert>
#include <iostream>

#include "grad_check.h"

int main() {
    using llm::Tensor;
    // matmul_grad_a check: C = A*B, loss=sum(C), dA = ones*B^T
    Tensor A({2, 3}, 0.0f);
    A.data = {0.5f, -1.0f, 2.0f, 1.0f, 0.0f, -0.5f};
    Tensor B({3, 2}, 0.0f);
    B.data = {1.0f, 0.5f, -1.0f, 2.0f, 0.3f, -0.7f};
    Tensor C = A.matmul(B);
    Tensor dC({2, 2}, 1.0f);
    Tensor dA = A.matmul_grad_a(B, dC);
    float err = llm::test::max_fd_error(A, [&](const Tensor& Ap) { return Ap.matmul(B); }, dA);
    std::cout << "matmul_grad_a err=" << err << "\n";
    assert(err < 2e-2);
    // layernorm_backward grad_x check
    Tensor x({2, 4}, 0.0f);
    x.data = {1, 2, 3, 4, -1, 0, 1, 2};
    Tensor gamma({4}, 1.0f);
    gamma.data = {0.5f, 1.0f, 1.5f, 2.0f};
    Tensor gout({2, 4}, 1.0f);
    auto lg = x.layernorm_backward(gout, &gamma, 1e-5f);
    float err2 = llm::test::max_fd_error(
        x, [&](const Tensor& xp) { return xp.layernorm(&gamma, nullptr); }, lg.grad_x);
    std::cout << "layernorm_backward err=" << err2 << "\n";
    assert(err2 < 2e-2);
    // rmsnorm_backward grad_x check via harness
    Tensor w({4}, 1.0f);
    w.data = gamma.data;
    auto rg = x.rmsnorm_backward(gout, &w, 1e-5f);
    float err3 =
        llm::test::max_fd_error(x, [&](const Tensor& xp) { return xp.rmsnorm(&w); }, rg.grad_x);
    std::cout << "rmsnorm_backward err=" << err3 << "\n";
    assert(err3 < 2e-2);
    std::cout << "grad_check test passed\n";
    return 0;
}
