// Regression: Tensor::transpose must materialize views in logical order
// (from_ndarray used to raw-copy shared storage, silently un-transposing).
#include "llm/tensor.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
    llm::Tensor K({4, 3}, 0.0f);
    for (size_t i = 0; i < 4; ++i) for (size_t j = 0; j < 3; ++j) K(i, j) = (float)(i * 10 + j);
    llm::Tensor Kt = K.transpose();
    assert(Kt.shape[0] == 3 && Kt.shape[1] == 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) assert(Kt(i, j) == (float)(j * 10 + i));
    // double transpose == identity
    llm::Tensor K2 = Kt.transpose();
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 3; ++j) assert(K2(i, j) == K(i, j));
    // matmul through a transposed operand matches scalar reference
    llm::Tensor Q({2, 3}, 0.0f);
    for (size_t i = 0; i < 6; ++i) Q.data[i] = (float)i * 0.5f - 1.0f;
    llm::Tensor S = Q.matmul(Kt);  // [2,4]
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j) {
            double acc = 0;
            for (size_t k = 0; k < 3; ++k) acc += (double)Q(i, k) * K(j, k);
            assert(std::fabs(S(i, j) - acc) < 1e-4);
        }
    std::cout << "transpose test passed\n";
    return 0;
}
