#include <cassert>
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
    assert(model.num_parameters() > 0);

    // Forward pass
    std::vector<int> tokens = {1, 2, 3, 4};
    auto logits = model.forward(tokens);
    assert(logits.shape[0] == 4 && logits.shape[1] == 32);

    // Check no NaN
    for (size_t i = 0; i < logits.data.size(); ++i) {
        assert(!std::isnan(logits.data[i]));
    }

    // Generate
    auto out = model.generate(tokens, 4, 0.0f, 0);
    assert(out.size() == 8);  // 4 prompt + 4 new

    std::cout << "test_mamba_model passed (params=" << model.num_parameters() << ")\n";
    return 0;
}
