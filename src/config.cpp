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
    if(cfg.n_embd % cfg.n_heads !=0) return false;
    if(cfg.vocab_size==0 || cfg.n_layers==0) return false;
    // C24: ALiBi and RoPE are mutually exclusive position encodings
    if(cfg.use_alibi && cfg.pos_encoding == PosEncoding::RoPE) return false;
    if(cfg.rope_scaling < 1.0f) return false;
    if(cfg.global_every == 0) return false;
    return true;
}
}
