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
#include "llm/gptq.h"
#include <cmath>
#include <iostream>
#include <string>

// F60: --bits {4,8} --group N --out path. Applies in-place simulated quantization
// (8-bit: int8 levels; 4-bit: GPTQ group levels, dequantized back to fp weights)
// and reports mean abs param delta as dequant verify.
static void apply_quantize_flags(llm::GPT& model, int bits, size_t group) {
    double sum = 0;
    size_t n = 0;
    if (bits == 4) {
        for (auto* p : model.parameters()) {
            if (p->shape.size() != 2) continue;
            llm::Tensor scale;
            llm::Tensor q = llm::quantize_4bit(*p, scale, group);
            llm::Tensor rec = llm::dequantize_4bit(q, scale, group);
            for (size_t i = 0; i < p->data.size(); ++i) sum += std::abs((*p).data[i] - rec.data[i]);
            n += p->data.size();
            *p = rec;
        }
    } else {
        for (auto* p : model.parameters()) {
            llm::QuantizedTensor qt = llm::quantize_with_scale(*p);
            llm::Tensor rec = llm::dequantize(qt);
            for (size_t i = 0; i < p->data.size(); ++i) sum += std::abs((*p).data[i] - rec.data[i]);
            n += p->data.size();
            *p = rec;
        }
    }
    std::cout << "[quantize] bits=" << bits << " group=" << group
              << " mean_abs_delta=" << (n ? sum / n : 0) << "\n";
}

void print_usage(const char* prog) {
    std::cout << "llm-cpp v" << LLM_CPP_VERSION << "\n";
    std::cout << "Usage: " << prog << " [train|generate] [options]\n"
              << "  train    --config <path> --data <path> [--checkpoint <path>] [--quantize] [--bits 4|8] [--group 128] [--out <quant.bin>]\n"
              << "  generate --prompt <text> [--max_tokens 100] [--config <path>] [--checkpoint <path>] [--temperature 1.0] [--top_k 0] [--top_p 1.0] [--quantize] [--bits 4|8] [--group 128] [--out <quant.bin>]\n"
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
    float lr = std::stof(args.get("lr", "6e-4"));
    llm::AdamW optim(lr);
    llm::CosineScheduler sched(lr, 100, 5000);
    bool do_quant = args.has("quantize");
    // D40: real CLI flags (was hardcoded max_iters=10)
    llm::TrainConfig tcfg;
    tcfg.max_iters = std::stoul(args.get("max_iters", "10"));
    tcfg.batch_size = std::stoul(args.get("batch_size", "32"));
    tcfg.eval_interval = std::stoul(args.get("eval_every", "500"));
    tcfg.grad_accum_steps = std::stoul(args.get("grad_accum", "1"));
    if (args.has("allow-untrained-params")) tcfg.allow_untrained_params = true;
    llm::Trainer trainer(model, tcfg, optim, sched);
    // D39: optional eval split during training
    std::string eval_data = args.get("eval_data", "");
    if (!eval_data.empty()) {
        llm::Dataset vds(eval_data, cfg.block_size);
        trainer.train(ds, &vds);
    } else trainer.train(ds);
    if (do_quant) {
        int bits = std::stoi(args.get("bits", "8"));
        size_t group = std::stoul(args.get("group", "128"));
        if (bits != 4 && bits != 8) { std::cerr << "[train] --bits must be 4 or 8\n"; return 1; }
        apply_quantize_flags(model, bits, group);
    }
    std::string ckpt = args.get("checkpoint", "checkpoints/model.bin");
    std::string qout = args.get("out", "");
    trainer.save_checkpoint(ckpt);
    if (do_quant && !qout.empty() && qout != ckpt) {
        // re-save quantized weights to --out (ckpt above already has them; copy path)
        model.save(qout);
        std::cout << "[train] quantized copy saved to " << qout << "\n";
    }
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
        int bits = std::stoi(args.get("bits", "8"));
        size_t group = std::stoul(args.get("group", "128"));
        if (bits != 4 && bits != 8) { std::cerr << "[generate] --bits must be 4 or 8\n"; return 1; }
        apply_quantize_flags(model, bits, group);
        std::string qout = args.get("out", "");
        if (!qout.empty()) { model.save(qout); std::cout << "[generate] quantized saved to " << qout << "\n"; }
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
