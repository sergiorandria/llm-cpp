#include <cmath>
#include <iostream>
#include "llm/checkpoint.h"
#include "llm/gguf.h"
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

    // Checkpoint v4 roundtrip: save → load → verify same output
    std::string ckpt_path = "/tmp/test_mamba_ckpt_v4.bin";
    model.save_binary(ckpt_path);

    llm::GPT model2(cfg);
    model2.load_binary(ckpt_path);

    auto logits2 = model2.forward(tokens);
    if (logits2.shape != logits.shape) {
        std::cerr << "FAIL: loaded logits shape mismatch\n";
        return 1;
    }
    for (size_t i = 0; i < logits.data.size(); ++i) {
        if (std::abs(logits.data[i] - logits2.data[i]) > 1e-6f) {
            std::cerr << "FAIL: logits mismatch at index " << i
                      << " orig=" << logits.data[i] << " loaded=" << logits2.data[i] << "\n";
            return 1;
        }
    }

    // Also verify generate produces same tokens
    auto out2 = model2.generate(tokens, 4, 0.0f, 0);
    if (out2.size() != out.size()) {
        std::cerr << "FAIL: generate output size mismatch after load\n";
        return 1;
    }
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] != out2[i]) {
            std::cerr << "FAIL: generate token mismatch at " << i
                      << " orig=" << out[i] << " loaded=" << out2[i] << "\n";
            return 1;
        }
    }

    // GGUF roundtrip (generic over parameters(), covers mamba blocks)
    {
        const std::string path = "/tmp/test_mamba_model.gguf";
        if (!llm::save_gguf(model, path)) {
            std::cerr << "FAIL: save_gguf mamba\n";
            return 1;
        }
        llm::GPT mg(cfg);
        if (!llm::load_gguf(mg, path)) {
            std::cerr << "FAIL: load_gguf mamba\n";
            return 1;
        }
        auto lg = mg.forward(tokens);
        float md = 0;
        for (size_t i = 0; i < logits.data.size(); ++i)
            md = std::max(md, std::abs(logits.data[i] - lg.data[i]));
        if (md > 1e-5f) {
            std::cerr << "FAIL: gguf mamba diff " << md << "\n";
            return 1;
        }
    }

    // SafeTensors roundtrip (gpt_named_params is generic over blocks)
    {
        auto n1 = llm::gpt_named_params(model);
        llm::save_safetensors("/tmp/test_mamba_model.safetensors", n1);
        llm::GPT ms(cfg);
        for (auto p : ms.parameters())
            for (auto& v : p->data) v += 1.0f;
        auto n2 = llm::gpt_named_params(ms);
        llm::load_safetensors("/tmp/test_mamba_model.safetensors", n2);
        auto ls = ms.forward(tokens);
        float md = 0;
        for (size_t i = 0; i < logits.data.size(); ++i)
            md = std::max(md, std::abs(logits.data[i] - ls.data[i]));
        if (md > 1e-6f) {
            std::cerr << "FAIL: safetensors mamba diff " << md << "\n";
            return 1;
        }
    }

    std::cout << "test_mamba_model passed (params=" << model.num_parameters() << ")\n";
    return 0;
}
