#include "llm/config.h"
#include <fstream>
#include <iostream>
namespace llm {
Config load_config(const std::string& path){
    Config cfg;
    std::ifstream in(path);
    if(!in){ std::cerr<<"[config] cannot open "<<path<<" using defaults\n"; return cfg; }
    // naive JSON parse: search for keys
    std::string content((std::istreambuf_iterator<char>(in)), {});
    auto get = [&](const std::string& key, size_t def){ size_t p=content.find(key); if(p==std::string::npos) return def; p=content.find(":",p); if(p==std::string::npos) return def; return (size_t)std::stoi(content.substr(p+1)); };
    cfg.vocab_size = get("\"vocab_size\"", cfg.vocab_size);
    cfg.n_layers = get("\"n_layers\"", cfg.n_layers);
    cfg.n_heads = get("\"n_heads\"", cfg.n_heads);
    cfg.n_embd = get("\"n_embd\"", cfg.n_embd);
    cfg.block_size = get("\"block_size\"", cfg.block_size);
    return cfg;
}
void save_config(const Config& cfg, const std::string& path){
    std::ofstream out(path);
    out<<"{\n  \"vocab_size\": "<<cfg.vocab_size<<",\n  \"n_layers\": "<<cfg.n_layers<<",\n  \"n_heads\": "<<cfg.n_heads<<",\n  \"n_embd\": "<<cfg.n_embd<<",\n  \"block_size\": "<<cfg.block_size<<"\n}\n";
}
bool validate_config(const Config& cfg){
    std::string err;
    return validate_config_verbose(cfg, err);
}
bool validate_config_verbose(const Config& cfg, std::string& err){
    if(cfg.n_embd % cfg.n_heads !=0) { err = "n_heads (" + std::to_string(cfg.n_heads) + ") must divide n_embd (" + std::to_string(cfg.n_embd) + ")"; return false; }
    if(cfg.vocab_size==0) { err = "vocab_size must be > 0"; return false; }
    if(cfg.n_layers==0) { err = "n_layers must be > 0"; return false; }
    // C24: ALiBi and RoPE are mutually exclusive position encodings
    if(cfg.use_alibi && cfg.pos_encoding == PosEncoding::RoPE) { err = "use_alibi and RoPE are mutually exclusive (pick one position encoding)"; return false; }
    if(cfg.rope_scaling < 1.0f) { err = "rope_scaling must be >= 1.0"; return false; }
    if(cfg.global_every == 0) { err = "global_every must be > 0"; return false; }
    err.clear();
    return true;
}
Config load_hf_config(const std::string& path){
    // HF GPT-2 style: {"n_layer":12,"n_head":12,"n_embd":768,"n_positions":1024,"n_vocab":50257,
    //   "activation_function":"gelu_new", ...} — unknown keys ignored.
    Config cfg;
    std::ifstream in(path);
    if(!in){ std::cerr<<"[config] cannot open "<<path<<" using defaults\n"; return cfg; }
    std::string c((std::istreambuf_iterator<char>(in)), {});
    auto get = [&](const std::string& key, size_t def){
        size_t p=c.find("\""+key+"\""); if(p==std::string::npos) return def;
        p=c.find(":",p); if(p==std::string::npos) return def;
        return (size_t)std::stoi(c.substr(p+1));
    };
    cfg.n_layers = get("n_layer", cfg.n_layers);
    size_t nh = get("n_head", cfg.n_heads);
    size_t kv = get("n_head_kv", nh); // GQA extension; ignored except validation below
    (void)kv;
    cfg.n_heads = nh;
    cfg.n_embd = get("n_embd", cfg.n_embd);
    cfg.block_size = get("n_positions", cfg.block_size);
    cfg.block_size = get("n_ctx", cfg.block_size);
    cfg.vocab_size = get("n_vocab", cfg.vocab_size);
    cfg.vocab_size = get("vocab_size", cfg.vocab_size);
    return cfg;
}
}
