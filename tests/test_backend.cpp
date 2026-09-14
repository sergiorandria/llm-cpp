// I87: backend probe + matmul correctness on the active backend.
// CPU build: asserts cpu/no-CUDA (unchanged). CUDA build: asserts cuda when a
// device is present (graceful CPU fallback otherwise) and checks cuBLAS SGEMM
// parity vs a naive reference on a 128x128 problem (above the offload floor).
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#include "llm/backend.h"
#include "llm/tensor.h"
int main() {
    llm::Tensor A({2, 2}, 0.0f), B({2, 2}, 0.0f);
    A.data = {1, 2, 3, 4};
    B.data = {5, 6, 7, 8};
    auto C = A.matmul(B);
    assert(C.data[0] == 19 && C.data[3] == 50);
#ifdef USE_CUDA
    if (llm::cuda_available()) {
        assert(std::string(llm::backend_name()) == "cuda");
        assert(llm::active_backend() == llm::Backend::CUDA);
        // 128^3 = 2M MACs > offload floor: must have run on cuBLAS.
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        const size_t M = 128, K = 128, N = 128;
        llm::Tensor X({M, K}, 0.0f), Y({K, N}, 0.0f);
        for (auto& v : X.data) v = dist(rng);
        for (auto& v : Y.data) v = dist(rng);
        llm::Tensor Z = X.matmul(Y);
        double max_err = 0;
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j) {
                double ref = 0;
                for (size_t k = 0; k < K; ++k) ref += (double)X.data[i * K + k] * Y.data[k * N + j];
                max_err = std::max(max_err, std::abs(ref - (double)Z.data[i * N + j]));
            }
        std::cout << "backend test passed (cuda, max_err=" << max_err << ")\n";
        assert(max_err < 1e-3);
    } else {
        // CUDA build, no device: honest CPU fallback must still be correct.
        assert(std::string(llm::backend_name()) == "cpu");
        std::cout << "backend test passed (cuda build, no device -> cpu fallback)\n";
    }
#else
    assert(!llm::cuda_available());
    assert(std::string(llm::backend_name()) == "cpu");
    assert(llm::active_backend() == llm::Backend::CPU);
    std::cout << "backend test passed (" << llm::backend_name() << ")\n";
#endif
    return 0;
}
