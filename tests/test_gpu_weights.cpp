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
    // Shapes must survive the roundtrip (2D stays 2D — old code flattened to 1D).
    if (out[0].shape.size() != 2 || out[0].shape[0] != 4 || out[0].shape[1] != 4) {
        std::cerr << "FAIL: out[0] shape not preserved\n";
        return 1;
    }
    if (out[1].shape.size() != 1 || out[1].shape[0] != 4) {
        std::cerr << "FAIL: out[1] shape not preserved\n";
        return 1;
    }
    for (float v : out[0].data)
        if (v != 1.0f) {
            std::cerr << "FAIL: expected 1.0f, got " << v << "\n";
            return 1;
        }
    for (float v : out[1].data)
        if (v != 2.0f) {
            std::cerr << "FAIL: expected 2.0f, got " << v << "\n";
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
