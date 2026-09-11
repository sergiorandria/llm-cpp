#include "llm/checkpoint.h"
#include "llm/model.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
namespace llm {
Tensor checkpointed_forward(const Tensor& x, const CheckpointConfig& cfg){
    if(!cfg.enabled) return x;
    // Stub: in real autograd we'd not stash activations, here we just return x
    // to demonstrate API; memory saving would be O(segment) vs O(layers)
    (void)cfg;
    return x;
}

// ── D32 SafeTensors F32 ──
void save_safetensors(const std::string& path,
                      const std::vector<std::pair<std::string, Tensor*>>& named) {
    std::ostringstream js;
    js << "{\"__metadata__\":{\"format\":\"llm-cpp-f32\"}";
    size_t off = 0;
    for (auto& kv : named) {
        js << ",\"" << kv.first << "\":{\"dtype\":\"F32\",\"shape\":[";
        for (size_t i = 0; i < kv.second->shape.size(); ++i) {
            if (i) js << ",";
            js << kv.second->shape[i];
        }
        size_t nbytes = kv.second->data.size() * sizeof(float);
        js << "],\"data_offsets\":[" << off << "," << off + nbytes << "]}";
        off += nbytes;
    }
    js << "}";
    std::string header = js.str();
    std::ofstream out(path, std::ios::binary);
    uint64_t hlen = header.size();
    out.write((char*)&hlen, 8);
    out.write(header.data(), header.size());
    for (auto& kv : named) out.write((char*)kv.second->data.data(),
                                     kv.second->data.size() * sizeof(float));
}

static std::string read_json_string(const std::string& j, size_t& i) {
    // j[i] == '"', returns unescaped (no escapes used in our names)
    ++i;
    std::string o;
    while (i < j.size() && j[i] != '"') o += j[i++];
    ++i;
    return o;
}

void load_safetensors(const std::string& path,
                      const std::vector<std::pair<std::string, Tensor*>>& named) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "[safetensors] cannot open " << path << "\n"; return; }
    uint64_t hlen = 0;
    in.read((char*)&hlen, 8);
    std::string header(hlen, '\0');
    in.read(header.data(), hlen);
    // parse entries: "name":{"dtype":"F32","shape":[...],"data_offsets":[s,e]}
    struct Ent { std::vector<size_t> shape; size_t s, e; };
    std::vector<std::pair<std::string, Ent>> ents;
    size_t i = 0;
    while (i < header.size()) {
        size_t q = header.find('"', i);
        if (q == std::string::npos) break;
        i = q;
        std::string key = read_json_string(header, i);
        // skip to '{' or ',' / '}'
        while (i < header.size() && header[i] != '{' && header[i] != ',' && header[i] != '}') ++i;
        if (i >= header.size() || header[i] != '{') continue;
        ++i; // into object (or __metadata__)
        if (key == "__metadata__") { int depth = 1; while (i < header.size() && depth) { if (header[i]=='{') ++depth; else if (header[i]=='}') --depth; ++i; } continue; }
        // find shape [...]
        auto ps = header.find('[', i);
        auto pe = header.find(']', ps);
        std::vector<size_t> shape;
        {
            std::string arr = header.substr(ps + 1, pe - ps - 1);
            std::stringstream ss(arr);
            std::string tok;
            while (std::getline(ss, tok, ',')) if (!tok.empty()) shape.push_back(std::stoul(tok));
        }
        auto po = header.find('[', pe + 1);
        auto po2 = header.find(']', po);
        std::string offs = header.substr(po + 1, po2 - po - 1);
        size_t s = std::stoul(offs.substr(0, offs.find(',')));
        size_t e = std::stoul(offs.substr(offs.find(',') + 1));
        // skip to end of entry object
        i = header.find('}', po2) + 1;
        ents.emplace_back(key, Ent{shape, s, e});
    }
    // read data section into memory
    std::string data((std::istreambuf_iterator<char>(in)), {});
    for (auto& kv : named) {
        for (auto& en : ents) {
            if (en.first != kv.first) continue;
            if (en.second.shape != kv.second->shape) {
                std::cerr << "[safetensors] shape mismatch for " << kv.first << "\n";
                break;
            }
            size_t n = kv.second->data.size() * sizeof(float);
            if (en.second.e - en.second.s != n || en.second.e > data.size()) {
                std::cerr << "[safetensors] offset mismatch for " << kv.first << "\n";
                break;
            }
            memcpy(kv.second->data.data(), data.data() + en.second.s, n);
            break;
        }
    }
}

std::vector<std::pair<std::string, Tensor*>> gpt_named_params(GPT& m) {
    std::vector<std::pair<std::string, Tensor*>> out;
    auto ps = m.parameters();
    std::vector<std::string> names = {"wte", "wpe", "ln_f_gamma", "ln_f_beta"};
    if (!m.config().weight_tying) names.push_back("lm_head");
    size_t n_blocks = m.config().n_layers;
    size_t idx = names.size();
    size_t nblk_params = n_blocks ? (ps.size() - idx) / n_blocks : 0;
    for (size_t b = 0; b < n_blocks; ++b)
        for (size_t j = 0; j < nblk_params; ++j)
            names.push_back("blk" + std::to_string(b) + ".p" + std::to_string(j));
    for (size_t i = 0; i < ps.size(); ++i) out.emplace_back(names[i], ps[i]);
    return out;
}
}
