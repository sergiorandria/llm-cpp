#include <cmath>
#include <iostream>
#include "llm/model.h"

int main() {
    llm::Config cfg;
    cfg.vocab_size = 32;
    cfg.n_embd = 32;
    cfg.n_heads = 4;    // ignored when use_mamba=true
    cfg.n_layers = 2;
    cfg.block_size = 64;
    cfg.weight_tying = true;
    cfg.use_mamba = true;
    cfg.d_inner = 64;
    cfg.d_state = 4;
    cfg.dt_rank = 4;
    cfg.conv_kernel = 4;

    llm::GPT model(cfg);
    if (model.num_parameters() == 0) {
        std::cerr << "FAIL: model.num_parameters() == 0\n";
        return 1;
    }

    // Forward pass
    std::vector<int> tokens = {1, 2, 3, 4};
    auto logits = model.forward(tokens);
    if (logits.shape[0] != 4 || logits.shape[1] != 32) {
        std::cerr << "FAIL: logits shape mismatch: " << logits.shape[0] << "x" << logits.shape[1] << "\n";
        return 1;
    }

    // Check no NaN
    for (size_t i = 0; i < logits.data.size(); ++i) {
        if (std::isnan(logits.data[i])) {
            std::cerr << "FAIL: NaN in logits at index " << i << "\n";
            return 1;
        }
    }

    // Generate
    auto out = model.generate(tokens, 4, 0.0f, 0);
    if (out.size() != 8) {
        std::cerr << "FAIL: generate output size " << out.size() << " != 8\n";
        return 1;
    }

    std::cout << "test_mamba_model passed (params=" << model.num_parameters() << ")\n";
    return 0;
}
