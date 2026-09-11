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
#include "llm/server.h"
#include "llm/utils.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

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
    std::cout << "Usage: " << prog << " [train|generate|serve] [options]\n"
              << "  train    --config <path> --data <path> [--checkpoint <path>] [--quantize] [--bits 4|8] [--group 128] [--out <quant.bin>]\n"
              << "  generate --prompt <text> [--max_tokens 100] [--config <path>] [--checkpoint <path>] [--temperature 1.0] [--top_k 0] [--top_p 1.0] [--quantize] [--bits 4|8] [--group 128] [--out <quant.bin>] [--stream] [--stop a,b] [--logprobs] [--seed N] [--json]\n"
              << "  serve    --port 8080 [--config <path>] [--checkpoint <path>] [--max_concurrency 8]\n"
              << "  --help   Show this help\n";
}

int main(int argc, char* argv[]) {
    llm::init_threading();  // I81: honor LLM_THREADS / OMP_NUM_THREADS
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
    { std::string verr; if(!llm::validate_config_verbose(cfg, verr)) { std::cerr<<"[train] invalid config: "<<verr<<"\n"; return 1; } }
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
        { std::string verr; if(!llm::validate_config_verbose(cfg, verr)) { std::cerr<<"[generate] invalid config "<<cfg_path<<": "<<verr<<"\n"; return 1; } }
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
    // G66: --seed for reproducible sampling
    if (args.has("seed")) llm::set_global_seed((uint64_t)std::stoul(args.get("seed", "42")));
    // G66: --stop comma-separated strings -> truncate text at first occurrence
    std::vector<std::string> stops;
    if (args.has("stop")) {
        std::string s = args.get("stop", "");
        size_t p = 0;
        while (p <= s.size()) {
            size_t c = s.find(',', p);
            if (c == std::string::npos) c = s.size();
            if (c > p) stops.push_back(s.substr(p, c - p));
            p = c + 1;
        }
    }
    auto apply_stops = [&](std::string t) {
        for (auto& st : stops) {
            size_t f = t.find(st);
            if (f != std::string::npos) t.resize(f);
        }
        return t;
    };
    if (args.has("json") || args.has("logprobs")) {
        auto t0 = std::chrono::steady_clock::now();
        auto go = model.generate_with_logprobs(ids, max_tokens, temp, top_k, top_p);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::string text = apply_stops(tok.decode(std::vector<int>(go.tokens.begin() + ids.size(), go.tokens.end())));
        if (args.has("json")) {
            std::cout << "{\"text\":\"";
            for (char c : text) {
                if (c == '"') std::cout << "\\\"";
                else if (c == '\\') std::cout << "\\\\";
                else if (c == '\n') std::cout << "\\n";
                else std::cout << c;
            }
            std::cout << "\",\"tokens\":[";
            for (size_t i = ids.size(); i < go.tokens.size(); ++i) {
                if (i > ids.size()) std::cout << ",";
                std::cout << go.tokens[i];
            }
            std::cout << "],\"logprobs\":[";
            for (size_t i = 0; i < go.logprobs.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << go.logprobs[i];
            }
            std::cout << "],\"latency_ms\":" << ms << "}\n";
        } else {
            std::cout << text << "\n";
            std::cerr << "[generate] latency_ms=" << ms << " logprobs=" << go.logprobs.size() << "\n";
        }
        return 0;
    }
    if (args.has("stream")) {
        // G66: true per-token streaming with UTF-8-safe incremental decode
        std::string carry;
        model.generate_streaming(ids, max_tokens, [&](int t) {
            std::string chunk = tok.decode_incremental({t}, carry);
            std::cout << chunk << std::flush;
        });
        if (!carry.empty()) std::cout << carry;
        std::cout << "\n";
        return 0;
    }
    auto out = model.generate(ids, max_tokens, temp, top_k, top_p, rep);
    std::cout << apply_stops(tok.decode(out)) << "\n";
    return 0;
    } else if (cmd == "serve") {
    // G61/G70: OpenAI-compatible server with graceful shutdown
    auto args = llm::parse_args(argc, argv);
    int port = std::stoi(args.get("port", "8080"));
    std::string cfg_path = args.get("config", "");
    std::string ckpt = args.get("checkpoint", "");
    llm::Config cfg;
    if (!cfg_path.empty()) {
        cfg = llm::load_config(cfg_path);
        std::string err;
        if (!llm::validate_config_verbose(cfg, err)) { std::cerr << "[serve] invalid config: " << err << "\n"; return 1; }
    } else {
        cfg.vocab_size=256; cfg.n_layers=2; cfg.n_heads=4; cfg.n_embd=64; cfg.block_size=128;
    }
    llm::GPT model(cfg);
    if (!ckpt.empty()) model.load(ckpt);
    llm::Tokenizer tok(256);
    llm::ServerConfig scfg;
    scfg.port = port;
    scfg.max_concurrency = std::stoul(args.get("max_concurrency", "8"));
    llm::Server srv(model, tok, scfg);
    if (!srv.start()) { std::cerr << "[serve] bind failed on port " << port << "\n"; return 1; }
    std::cout << "[serve] listening on 127.0.0.1:" << srv.port() << " (max_concurrency=" << scfg.max_concurrency << ")\n";
    static std::atomic<bool> g_run{true};
    static llm::Server* g_srv = nullptr;
    g_srv = &srv;
    std::signal(SIGTERM, [](int){ g_run = false; if (g_srv) g_srv->stop(); });
    std::signal(SIGINT, [](int){ g_run = false; if (g_srv) g_srv->stop(); });
    // G70: block until signal; handler drains in-flight via stop()
    while (g_run.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    srv.stop();
    std::cout << "[serve] drained, bye\n";
    return 0;
    }

    print_usage(argv[0]);
    return 1;
}
