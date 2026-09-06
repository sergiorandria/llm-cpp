#include "llm/gguf.h"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace llm {

static constexpr uint32_t GGUF_MAGIC = 0x46554747u; // "GGUF" little-endian
static constexpr uint32_t GGUF_VERSION = 3;
static constexpr uint32_t GGUF_DTYPE_F32 = 0;
static constexpr size_t GGUF_ALIGN = 32;
static constexpr uint32_t GGUF_TYPE_STRING = 8;

static size_t align_offset(size_t offset, size_t align) {
    return (offset + align - 1) & ~(align - 1);
}

static void write_u32(std::ofstream& out, uint32_t v) { out.write((char*)&v, 4); }
static void write_u64(std::ofstream& out, uint64_t v) { out.write((char*)&v, 8); }
static void write_string(std::ofstream& out, const std::string& s) {
    uint64_t len = s.size();
    out.write((char*)&len, 8);
    if (len) out.write(s.data(), len);
}
static bool read_u32(std::ifstream& in, uint32_t& v) {
    in.read((char*)&v, 4);
    return !in.fail();
}
static bool read_u64(std::ifstream& in, uint64_t& v) {
    in.read((char*)&v, 8);
    return !in.fail();
}
static bool read_string(std::ifstream& in, std::string& s) {
    uint64_t len = 0;
    if (!read_u64(in, len)) return false;
    s.resize(len);
    if (len) {
        in.read(s.data(), len);
        if (in.fail()) return false;
    }
    return true;
}

bool save_gguf(const GPT& model, const std::string& path) {
    // Ensure parent directory exists
    {
        size_t slash = path.find_last_of("/\\");
        if (slash != std::string::npos) {
            std::string dir = path.substr(0, slash);
            std::filesystem::create_directories(dir);
        }
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "[gguf] save: cannot open " << path << "\n";
        return false;
    }

    // Collect parameters and generate names/shapes
    auto params = model.parameters(); // const overload: vector<const Tensor*>
    size_t tensor_count = params.size();

    // KV metadata — config + general
    struct KV { std::string key, value; };
    std::vector<KV> kvs;
    // Use model num_parameters to infer nothing, but store config fields explicitly
    // We need config values: vocab_size, n_layers, n_heads, n_embd, block_size, weight_tying
    // Since GPT doesn't expose config directly, we deduce from parameters shapes:
    //  wte shape [vocab, n_embd] -> vocab, n_embd
    //  wpe shape [block_size, n_embd] -> block_size
    //  n_layers = (tensor_count - fixed) / per_block
    // Instead store via model.num_parameters? Better to store derived and also store config via inspection.
    // We will store what we can: retrieve via parameters but also need original config.
    // For now, store heuristic: we serialize config by reading first tensors.
    // To avoid hack, we store actual config fields by using a small trick: we have access via private config_?
    // Since we are not friend, we derive n_embd from wte shape[1], vocab from wte shape[0], block_size from wpe shape[0].
    // n_layers we derive as blocks size: we can count blocks via tensor_count pattern, but we store it directly as kvs.
    // Alternative: we can store all params' shapes as sufficient to reconstruct; but we also store textual config for validation.
    // We'll compute:
    size_t vocab_size = 0, n_embd = 0, block_size = 0;
    if (tensor_count >= 2) {
        vocab_size = params[0]->shape.size() >= 2 ? params[0]->shape[0] : 0;
        n_embd = params[0]->shape.size() >= 2 ? params[0]->shape[1] : 0;
        block_size = params[1]->shape.size() >= 2 ? params[1]->shape[0] : 0;
    }
    // Estimate n_layers: each block has at least 8 tensors (attn 4 + ffn 2-5 + ln 4) -> ~10-14
    // We store tensor_count itself, so loader can validate shape list without needing n_layers.
    // Store generic KVs
    kvs.push_back({"general.architecture", "llm"});
    kvs.push_back({"general.name", "llm-cpp"});
    kvs.push_back({"llm.vocab_size", std::to_string(vocab_size)});
    kvs.push_back({"llm.n_embd", std::to_string(n_embd)});
    kvs.push_back({"llm.block_size", std::to_string(block_size)});
    kvs.push_back({"llm.tensor_count", std::to_string(tensor_count)});

    uint64_t kv_count = kvs.size();

    // Header
    write_u32(out, GGUF_MAGIC);
    write_u32(out, GGUF_VERSION);
    write_u64(out, (uint64_t)tensor_count);
    write_u64(out, kv_count);

    // KV section
    for (auto& kv : kvs) {
        write_string(out, kv.key);
        write_u32(out, GGUF_TYPE_STRING);
        write_string(out, kv.value);
    }

    // Prepare tensor infos with names "tensor_0", "tensor_1", ...
    struct Info {
        std::string name;
        std::vector<uint64_t> dims;
        uint32_t dtype = GGUF_DTYPE_F32;
        uint64_t offset = 0;
        uint64_t nbytes = 0;
    };
    std::vector<Info> infos;
    infos.reserve(tensor_count);
    for (size_t i = 0; i < tensor_count; ++i) {
        Info info;
        info.name = "tensor_" + std::to_string(i);
        info.dims.reserve(params[i]->shape.size());
        for (auto d : params[i]->shape) info.dims.push_back((uint64_t)d);
        info.dtype = GGUF_DTYPE_F32;
        info.nbytes = params[i]->data.size() * sizeof(float);
        infos.push_back(std::move(info));
    }

    // Compute offsets aligned to 32, relative to data section start
    size_t current_offset = 0;
    for (auto& info : infos) {
        current_offset = align_offset(current_offset, GGUF_ALIGN);
        info.offset = current_offset;
        current_offset += info.nbytes;
    }

    // Write tensor infos
    for (auto& info : infos) {
        write_string(out, info.name);
        uint32_t n_dims = (uint32_t)info.dims.size();
        write_u32(out, n_dims);
        for (auto d : info.dims) write_u64(out, d);
        write_u32(out, info.dtype);
        write_u64(out, info.offset);
    }

    // Align to 32 before data section
    size_t header_end = (size_t)out.tellp();
    size_t aligned = align_offset(header_end, GGUF_ALIGN);
    size_t pad = aligned - header_end;
    for (size_t i = 0; i < pad; ++i) out.put(0);

    // Write tensor data at computed offsets
    // Since we already computed offsets sequentially aligned, we just write with padding
    size_t data_start = (size_t)out.tellp();
    size_t written = 0;
    for (size_t i = 0; i < infos.size(); ++i) {
        // Pad to align
        size_t target = data_start + infos[i].offset;
        size_t cur = (size_t)out.tellp();
        if (cur < target) {
            size_t need = target - cur;
            for (size_t p = 0; p < need; ++p) out.put(0);
        } else if (cur > target) {
            std::cerr << "[gguf] save: offset miscalc\n";
            return false;
        }
        const auto* t = params[i];
        out.write((char*)t->data.data(), t->data.size() * sizeof(float));
        written += t->data.size() * sizeof(float);
    }

    out.flush();
    if (!out) {
        std::cerr << "[gguf] save: write failed " << path << "\n";
        return false;
    }
    std::cout << "[gguf] save " << path << " tensors " << tensor_count << " kvs " << kv_count
              << " bytes " << written << " aligned " << GGUF_ALIGN << "\n";
    return true;
}

bool load_gguf(GPT& model, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[gguf] load: cannot open " << path << "\n";
        return false;
    }
    uint32_t magic = 0, version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (!read_u32(in, magic) || !read_u32(in, version) || !read_u64(in, tensor_count) ||
        !read_u64(in, kv_count)) {
        std::cerr << "[gguf] load: header read failed\n";
        return false;
    }
    if (magic != GGUF_MAGIC) {
        std::cerr << "[gguf] load: bad magic 0x" << std::hex << magic << " expected 0x"
                  << GGUF_MAGIC << std::dec << "\n";
        return false;
    }
    if (version != GGUF_VERSION) {
        std::cerr << "[gguf] load: unsupported version " << version << " expected " << GGUF_VERSION
                  << "\n";
        return false;
    }

    // Read KV
    std::vector<std::pair<std::string, std::string>> kvs;
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key, value;
        uint32_t type = 0;
        if (!read_string(in, key) || !read_u32(in, type) || !read_string(in, value)) {
            std::cerr << "[gguf] load: kv read failed at " << i << "\n";
            return false;
        }
        if (type != GGUF_TYPE_STRING) {
            std::cerr << "[gguf] load: unexpected kv type " << type << "\n";
            return false;
        }
        kvs.emplace_back(key, value);
    }

    // Collect expected config from KV for validation (optional)
    // We have vocab_size, n_embd, block_size stored; we can validate against current model
    auto mutable_params = model.parameters(); // non-const for writing
    if (mutable_params.size() != tensor_count) {
        std::cerr << "[gguf] load: tensor count mismatch file " << tensor_count << " vs model "
                  << mutable_params.size() << " — config mismatch, skipping\n";
        return false;
    }

    struct Info {
        std::string name;
        std::vector<uint64_t> dims;
        uint32_t dtype = 0;
        uint64_t offset = 0;
    };
    std::vector<Info> infos;
    infos.reserve(tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        Info info;
        if (!read_string(in, info.name)) {
            std::cerr << "[gguf] load: tensor name read failed " << i << "\n";
            return false;
        }
        uint32_t n_dims = 0;
        if (!read_u32(in, n_dims)) {
            std::cerr << "[gguf] load: n_dims read failed\n";
            return false;
        }
        info.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim = 0;
            if (!read_u64(in, dim)) {
                std::cerr << "[gguf] load: dim read failed\n";
                return false;
            }
            info.dims[d] = dim;
        }
        if (!read_u32(in, info.dtype) || !read_u64(in, info.offset)) {
            std::cerr << "[gguf] load: dtype/offset read failed\n";
            return false;
        }
        if (info.dtype != GGUF_DTYPE_F32) {
            std::cerr << "[gguf] load: unsupported dtype " << info.dtype << "\n";
            return false;
        }
        infos.push_back(std::move(info));
    }

    // Data section start aligned to 32
    size_t cur_pos = (size_t)in.tellg();
    size_t data_start = align_offset(cur_pos, GGUF_ALIGN);

    // Validate shapes before mutating model
    for (size_t i = 0; i < infos.size(); ++i) {
        const auto& info = infos[i];
        auto* p = mutable_params[i];
        // Compare dims
        if (info.dims.size() != p->shape.size()) {
            std::cerr << "[gguf] load: shape ndim mismatch tensor " << info.name << " file "
                      << info.dims.size() << " vs model " << p->shape.size() << "\n";
            return false;
        }
        for (size_t d = 0; d < info.dims.size(); ++d) {
            if ((size_t)info.dims[d] != p->shape[d]) {
                std::cerr << "[gguf] load: shape dim mismatch tensor " << info.name << " dim " << d
                          << " file " << info.dims[d] << " vs model " << p->shape[d] << "\n";
                return false;
            }
        }
        // Validate offset + nbytes within file
        size_t nbytes = p->data.size() * sizeof(float);
        // We'll check later by seeking, but ensure offset+ nbytes doesn't overflow
        if (info.offset + nbytes < info.offset) {
            std::cerr << "[gguf] load: offset overflow\n";
            return false;
        }
    }

    // Now populate tensors
    for (size_t i = 0; i < infos.size(); ++i) {
        const auto& info = infos[i];
        auto* p = mutable_params[i];
        size_t nbytes = p->data.size() * sizeof(float);
        size_t abs_offset = data_start + (size_t)info.offset;
        in.seekg((std::streampos)abs_offset, std::ios::beg);
        if (in.fail()) {
            std::cerr << "[gguf] load: seek failed for " << info.name << " offset " << abs_offset
                      << "\n";
            return false;
        }
        in.read((char*)p->data.data(), nbytes);
        if ((size_t)in.gcount() != nbytes || in.fail()) {
            std::cerr << "[gguf] load: data read failed for " << info.name << " expected " << nbytes
                      << " got " << in.gcount() << "\n";
            return false;
        }
        // Ensure strides recomputed if shape changed (they didn't, but recompute for safety)
        // Tensor strides are computed in ctor; we can keep as is since shape unchanged.
    }

    // If weight tying, ensure lm_head reflects wte
    model.tie_weights();

    std::cout << "[gguf] load " << path << " tensors " << tensor_count << " kvs " << kv_count
              << " ok\n";
    return true;
}

} // namespace llm
