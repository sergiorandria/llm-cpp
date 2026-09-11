// I81: LLM_THREADS honored. I82: accelerated matmul == naive reference.
#include "llm/tensor.h"
#include "llm/utils.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
int main() {
    // I81
    setenv("LLM_THREADS", "2", 1);
    int nthr = llm::init_threading();
    std::cout << "threads=" << nthr << " caps=" << llm::simd_caps() << "\n";
#ifdef _OPENMP
    assert(nthr == 2);
#else
    assert(nthr == 1);
#endif
    unsetenv("LLM_THREADS");
    // I82: reference naive triple-loop vs Tensor::matmul (numpy-cpp/SIMD path)
    llm::Tensor A({16, 16}, 0.0f), B({16, 16}, 0.0f);
    for (size_t i = 0; i < A.data.size(); ++i) A.data[i] = (float)(i % 13) * 0.07f - 0.4f;
    for (size_t i = 0; i < B.data.size(); ++i) B.data[i] = (float)(i % 11) * 0.09f - 0.3f;
    llm::Tensor C = A.matmul(B);
    float md = 0;
    for (size_t i = 0; i < 16; ++i)
        for (size_t j = 0; j < 16; ++j) {
            double acc = 0;
            for (size_t k = 0; k < 16; ++k) acc += (double)A(i, k) * B(k, j);
            md = std::max(md, (float)std::fabs(C(i, j) - acc));
        }
    std::cout << "simd parity maxd=" << md << "\n";
    assert(md < 1e-3f);
    assert(!llm::simd_caps().empty());
    std::cout << "threads+simd test passed\n";
    return 0;
}
