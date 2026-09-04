#include "llm/tensor.h"
#ifdef USE_NUMPY_CPP
#include <np/np.hpp>
#include <np/linalg.hpp>
#include <np/random.hpp>
#include "llm/numpy_utils.h"
#endif
#include <iostream>

int main() {
    std::cout << "=== numpy-cpp backend demo ===\n";
#ifdef USE_NUMPY_CPP
    std::cout << "USE_NUMPY_CPP enabled (C++20)\n";
    // 1. Matmul via numpy-cpp blocked GEMM
    llm::Tensor A({2,3}, 1.0f);
    llm::Tensor B({3,2}, 2.0f);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    B(0,0)=4; B(1,0)=5; B(2,0)=6;
    auto C = A.matmul(B); // dispatches to np::linalg::matmul
    C.print("C = A @ B");

    // 2. Direct numpy interop
    auto np_a = A.to_ndarray();
    auto np_b = B.to_ndarray();
    auto np_c = np::linalg::matmul(np_a, np_b);
    std::cout << "np_c shape [" << np_c.shape[0] << "," << np_c.shape[1] << "] data[0]=" << np_c.data()[0] << "\n";

    // 3. Randn via np::random::Generator (PCG64)
    llm::Tensor W({4,4});
    W.randn(0, 0.02f);
    W.print("W randn");

    // 4. Transpose via numpy view
    auto T = A.transpose();
    T.print("A^T");

    // 5. Statistics via numpy_utils
    float m = llm::numpy_utils::mean_np(W);
    std::cout << "mean(W)=" << m << "\n";

#else
    std::cout << "USE_NUMPY_CPP not enabled - fallback naive ops\n";
    llm::Tensor A({2,2}, 1.0f);
    A.print("A");
#endif
    return 0;
}
