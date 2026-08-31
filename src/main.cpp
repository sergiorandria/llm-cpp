#include "llm/model.h"
#include "llm/tokenizer.h"
#include <iostream>
#include <string>

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [train|generate] [options]\n"
              << "  train    --config <path> --data <path>\n"
              << "  generate --prompt <text> [--max_tokens 100]\n"
              << "  --help   Show this help\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        // Demo: tiny forward pass
        llm::Config cfg;
        cfg.vocab_size = 256;
        cfg.n_layers = 2;
        cfg.n_heads = 4;
        cfg.n_embd = 64;
        cfg.block_size = 32;
        llm::GPT model(cfg);
        llm::Tokenizer tok(256);
        std::string prompt = "Hello LLM from C++!";
        auto ids = tok.encode(prompt);
        std::cout << "\n[demo] prompt: \"" << prompt << "\" -> ids size " << ids.size() << "\n";
        auto logits = model.forward(ids);
        std::cout << "[demo] logits shape [" << logits.shape[0] << ", " << logits.shape[1] << "]\n";
        std::cout << "[demo] params ~ " << model.num_parameters() << "\n";
        auto gen = model.generate(ids, 20);
        std::cout << "[demo] generated: \"" << tok.decode(gen) << "\"\n";
        return 0;
    }

    std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h") {
        print_usage(argv[0]);
        return 0;
    } else if (cmd == "train") {
        std::cout << "[train] not yet implemented — see README. Config: src/model.cpp\n";
        return 0;
    } else if (cmd == "generate") {
        llm::Config cfg;
        cfg.vocab_size = 256;
        cfg.n_layers = 2;
        cfg.n_heads = 4;
        cfg.n_embd = 64;
        cfg.block_size = 128;
        llm::GPT model(cfg);
        llm::Tokenizer tok(256);
        std::string prompt = argc > 3 ? argv[3] : "Hello";
        size_t max_tokens = 50;
        auto ids = tok.encode(prompt);
        auto out = model.generate(ids, max_tokens);
        std::cout << tok.decode(out) << "\n";
        return 0;
    }

    print_usage(argv[0]);
    return 1;
}
