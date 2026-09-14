#include <iostream>
#include "llm/gpu_weights.h"
#include "llm/tensor.h"

int main() {
    llm::GPUWeights gw;
    if (gw.count() != 0) {
        std::cerr << "FAIL: initial count should be 0\n";
        return 1;
    }

    std::vector<llm::Tensor> params;
    params.emplace_back(llm::Tensor({4, 4}, 1.0f));
    params.emplace_back(llm::Tensor({4}, 2.0f));

    gw.upload(params);
    if (gw.count() != 2) {
        std::cerr << "FAIL: count should be 2 after upload, got " << gw.count() << "\n";
        return 1;
    }

    // Download back
    std::vector<llm::Tensor> out;
    gw.download(out);
    if (out.size() != 2) {
        std::cerr << "FAIL: download size should be 2, got " << out.size() << "\n";
        return 1;
    }
    if (out[0].data[0] != 1.0f) {
        std::cerr << "FAIL: expected 1.0f, got " << out[0].data[0] << "\n";
        return 1;
    }
    if (out[1].data[0] != 2.0f) {
        std::cerr << "FAIL: expected 2.0f, got " << out[1].data[0] << "\n";
        return 1;
    }

    // Device pointer access (GPU only)
    if (gw.device_ptr(0)) {
        std::cout << "GPU ptr: " << gw.device_ptr(0) << "\n";
    }

    gw.clear();
    if (gw.count() != 0) {
        std::cerr << "FAIL: count should be 0 after clear\n";
        return 1;
    }

    std::cout << "test_gpu_weights passed\n";
    return 0;
}
