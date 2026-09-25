#include <cmath>
#include <iostream>
#include <vector>

#include "llm/model.h"

int main() {
    using namespace llm;

    // Test 1: GPT with use_gpu_weights=true uploads all params
    {
        Config cfg;
        cfg.vocab_size = 256;
        cfg.n_layers = 2;
        cfg.n_heads = 4;
        cfg.n_embd = 32;
        cfg.block_size = 16;
        cfg.use_mamba = true;
        cfg.d_inner = 64;
        cfg.d_state = 8;
        cfg.dt_rank = 4;
        cfg.conv_kernel = 4;
        cfg.use_gpu_weights = true;

        GPT model(cfg);
        size_t n_params = model.parameters().size();
        size_t n_gpu = model.gpu_weights_count();
        if (n_gpu != n_params) {
            std::cerr << "FAIL: gpu_weights count " << n_gpu << " != params " << n_params << "\n";
            return 1;
        }
        // Forward still works with upload on (weights are read from CPU mirror
        // on CPU builds; device pointers are used under USE_CUDA).
        auto logits = model.forward({1, 2, 3, 4});
        for (float v : logits.data)
            if (!std::isfinite(v)) {
                std::cerr << "FAIL: non-finite logits with gpu weights on\n";
                return 1;
            }
        // Save/load roundtrip preserves outputs with gpu weights on.
        model.save_binary("/tmp/test_gpu_integ.bin");
        GPT m2(cfg);
        m2.load_binary("/tmp/test_gpu_integ.bin");
        if (m2.gpu_weights_count() != n_params) {
            std::cerr << "FAIL: reloaded gpu_weights count mismatch\n";
            return 1;
        }
        auto l2 = m2.forward({1, 2, 3, 4});
        float md = 0;
        for (size_t i = 0; i < logits.data.size(); ++i)
            md = std::max(md, std::abs(logits.data[i] - l2.data[i]));
        if (md > 1e-6f) {
            std::cerr << "FAIL: reload diff " << md << "\n";
            return 1;
        }
        std::cout << "PASS: mamba gpu_weights uploaded " << n_params << " params\n";
    }

    // Test 2: GPT with use_gpu_weights=false does NOT upload
    {
        Config cfg;
        cfg.vocab_size = 256;
        cfg.n_layers = 2;
        cfg.n_heads = 4;
        cfg.n_embd = 32;
        cfg.block_size = 16;
        cfg.use_mamba = true;
        cfg.d_inner = 64;
        cfg.d_state = 8;
        cfg.dt_rank = 4;
        cfg.conv_kernel = 4;
        cfg.use_gpu_weights = false;

        GPT model(cfg);
        if (model.gpu_weights_count() != 0) {
            std::cerr << "FAIL: gpu_weights should be 0 when disabled\n";
            return 1;
        }
        std::cout << "PASS: GPT with use_gpu_weights=false\n";
    }

    // Test 3: Transformer path with use_gpu_weights
    {
        Config cfg;
        cfg.vocab_size = 256;
        cfg.n_layers = 2;
        cfg.n_heads = 4;
        cfg.n_embd = 32;
        cfg.block_size = 16;
        cfg.use_mamba = false;
        cfg.use_gpu_weights = true;

        GPT model(cfg);
        size_t n_params = model.parameters().size();
        if (model.gpu_weights_count() != n_params) {
            std::cerr << "FAIL: transformer gpu_weights count mismatch\n";
            return 1;
        }
        std::cout << "PASS: Transformer GPT with use_gpu_weights=true uploaded " << n_params
                  << " params\n";
    }

    std::cout << "All GPU integration tests passed\n";
    return 0;
}
