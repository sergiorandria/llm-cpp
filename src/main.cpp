#include "llm/model.h"
#include "llm/tokenizer.h"
#include "llm/config.h"
#include "llm/cli.h"
#include "llm/version.h"
#include "llm/dataset.h"
#include "llm/trainer.h"
#include "llm/optimizer.h"
#include "llm/scheduler.h"
#include "llm/quantize.h"
#include <iostream>
#include <string>

void print_usage(const char* prog) {
    std::cout << "llm-cpp v" << LLM_CPP_VERSION << "\n";
    std::cout << "Usage: " << prog << " [train|generate] [options]\n"
              << "  train    --config <path> --data <path> [--checkpoint <path>] [--quantize]\n"
              << "  generate --prompt <text> [--max_tokens 100] [--config <path>] [--checkpoint <path>] [--temperature 1.0] [--top_k 0] [--top_p 1.0] [--quantize]\n"
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
    auto args = llm::parse_args(argc, argv);
    std::string cfg_path = args.get("config", "config/config.json.example");
    std::string data_path = args.get("data", "data/input.txt");
    llm::Config cfg = llm::load_config(cfg_path);
    if(!llm::validate_config(cfg)) { std::cerr<<"[train] invalid config\n"; return 1; }
    std::cout << "[train] config "<<cfg_path<<" data "<<data_path<<" n_layers="<<cfg.n_layers<<"\n";
    llm::GPT model(cfg);
    std::cout<<"[train] model params "<<model.num_parameters()<<"\n";
    llm::Dataset ds(data_path, cfg.block_size);
    std::cout<<"[train] dataset tokens "<<ds.size()<<"\n";
    llm::AdamW optim(6e-4f);
    llm::CosineScheduler sched(6e-4f, 100, 5000);
    bool do_quant = args.has("quantize");
    llm::TrainConfig tcfg; tcfg.max_iters = 10; // demo: 10 steps so CLI doesn't hang
    llm::Trainer trainer(model, tcfg, optim, sched);
    trainer.train(ds);
    if (do_quant) {
        std::cout<<"[train] quantizing to int8...\n";
        quantize_model(model);
    }
    std::string ckpt = args.get("checkpoint", "checkpoints/model.bin");
    trainer.save_checkpoint(ckpt);
    std::cout<<"[train] done, saved "<<ckpt<<(do_quant?" (quantized)":"")<<"\n";
    return 0;
} else if (cmd == "generate") {
    auto args = llm::parse_args(argc, argv);
    std::string prompt = args.get("prompt", "Hello");
    float temp = std::stof(args.get("temperature", "1.0"));
    int top_k = std::stoi(args.get("top_k", "0"));
    float top_p = std::stof(args.get("top_p", "1.0"));
    float rep = std::stof(args.get("repetition_penalty", "1.0"));
    size_t max_tokens = std::stoi(args.get("max_tokens", "50"));
    std::string ckpt = args.get("checkpoint", "");
    std::string cfg_path = args.get("config", "");
    llm::Config cfg;
    if (!cfg_path.empty()) {
        cfg = llm::load_config(cfg_path);
        if(!llm::validate_config(cfg)) { std::cerr<<"[generate] invalid config "<<cfg_path<<"\n"; return 1; }
        std::cout<<"[generate] config "<<cfg_path<<"\n";
    } else {
        cfg.vocab_size=256; cfg.n_layers=2; cfg.n_heads=4; cfg.n_embd=64; cfg.block_size=128;
    }
    // Auto-infer vocab from checkpoint if available and no explicit config
    llm::GPT model(cfg);
    if(!ckpt.empty()){
        std::cout<<"[generate] loading "<<ckpt<<"\n";
        model.load(ckpt);
    }
    if (args.has("quantize")) {
        std::cout<<"[generate] quantizing model to int8 for inference\n";
        quantize_model(model);
    }
    llm::Tokenizer tok(256);
    auto ids = tok.encode(prompt);
    auto out = model.generate(ids, max_tokens, temp, top_k, top_p, rep);
    std::cout << tok.decode(out) << "\n";
    return 0;
    }

    print_usage(argv[0]);
    return 1;
}
