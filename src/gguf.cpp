#include "llm/gguf.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace llm {

static constexpr uint32_t GGUF_MAGIC = 0x46554747u;  // "GGUF" little-endian
static constexpr uint32_t GGUF_VERSION = 3;
static constexpr uint32_t GGUF_DTYPE_F32 = 0;
static constexpr uint32_t GGUF_DTYPE_Q4_0 = 2;  // ggml type id, block 32 nibbles + F32 scale
static constexpr uint32_t GGUF_DTYPE_Q8_0 = 8;  // ggml type id, block 32 int8 + F32 scale
static constexpr size_t GGUF_QBLK = 32;
static constexpr size_t GGUF_ALIGN = 32;
static constexpr uint32_t GGUF_TYPE_STRING = 8;

static size_t align_offset(size_t offset, size_t align) {
    return (offset + align - 1) & ~(align - 1);
}

static void write_u32(std::ofstream& out, uint32_t v) {
    out.write((char*)&v, 4);
}
static void write_u64(std::ofstream& out, uint64_t v) {
    out.write((char*)&v, 8);
}
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
    if (len > (1u << 20)) return false;  // H77: cap attacker-controlled alloc (DoS)
    s.resize(len);
    if (len) {
        in.read(s.data(), len);
        if (in.fail()) return false;
    }
    return true;
}

// ── F54 block codecs ──
void encode_q80_block(const float* x, float& scale_out, int8_t* q_out) {
    float mx = 0;
    for (size_t i = 0; i < GGUF_QBLK; ++i) mx = std::max(mx, std::abs(x[i]));
    scale_out = mx / 127.0f + 1e-8f;
    for (size_t i = 0; i < GGUF_QBLK; ++i) {
        float q = std::round(x[i] / scale_out);
        q_out[i] = (int8_t)std::max(-127.0f, std::min(127.0f, q));
    }
}
void decode_q80_block(float scale, const int8_t* q, float* out) {
    for (size_t i = 0; i < GGUF_QBLK; ++i) out[i] = (float)q[i] * scale;
}
void encode_q40_block(const float* x, float& scale_out, uint8_t* packed_out) {
    float mx = 0;
    for (size_t i = 0; i < GGUF_QBLK; ++i) mx = std::max(mx, std::abs(x[i]));
    scale_out = mx / 7.0f + 1e-8f;
    for (size_t i = 0; i < GGUF_QBLK; i += 2) {
        float a = std::round(x[i] / scale_out), b = std::round(x[i + 1] / scale_out);
        a = std::max(-8.0f, std::min(7.0f, a));
        b = std::max(-8.0f, std::min(7.0f, b));
        packed_out[i / 2] = (uint8_t)(((int)(b + 8) << 4) | ((int)(a + 8) & 0xF));
    }
}
void decode_q40_block(float scale, const uint8_t* packed, float* out) {
    for (size_t i = 0; i < GGUF_QBLK; i += 2) {
        out[i] = (float)(((int)(packed[i / 2] & 0xF)) - 8) * scale;
        out[i + 1] = (float)(((int)(packed[i / 2] >> 4) & 0xF) - 8) * scale;
    }
}

// Encoded size of a tensor with n floats under dtype tag
static size_t gguf_nbytes(size_t n, uint32_t dtype) {
    if (dtype == GGUF_DTYPE_Q8_0) return (n / GGUF_QBLK) * (4 + GGUF_QBLK);
    if (dtype == GGUF_DTYPE_Q4_0) return (n / GGUF_QBLK) * (4 + GGUF_QBLK / 2);
    return n * sizeof(float);
}

static void write_encoded(std::ofstream& out, const Tensor& t, uint32_t dtype) {
    size_t n = t.data.size();
    if (dtype == GGUF_DTYPE_F32) {
        out.write((char*)t.data.data(), n * sizeof(float));
        return;
    }
    std::vector<float> padded(t.data.begin(), t.data.end());
    while (padded.size() % GGUF_QBLK) padded.push_back(0.0f);
    if (dtype == GGUF_DTYPE_Q8_0) {
        for (size_t b = 0; b < padded.size(); b += GGUF_QBLK) {
            float sc;
            int8_t q[GGUF_QBLK];
            encode_q80_block(padded.data() + b, sc, q);
            out.write((char*)&sc, 4);
            out.write((char*)q, GGUF_QBLK);
        }
    } else {  // Q4_0
        for (size_t b = 0; b < padded.size(); b += GGUF_QBLK) {
            float sc;
            uint8_t p[GGUF_QBLK / 2];
            encode_q40_block(padded.data() + b, sc, p);
            out.write((char*)&sc, 4);
            out.write((char*)p, GGUF_QBLK / 2);
        }
    }
}

static bool read_decoded(std::ifstream& in, Tensor& t, uint32_t dtype, size_t abs_offset) {
    in.seekg((std::streampos)abs_offset, std::ios::beg);
    if (in.fail()) return false;
    size_t n = t.data.size();
    if (dtype == GGUF_DTYPE_F32) {
        in.read((char*)t.data.data(), n * sizeof(float));
        return (size_t)in.gcount() == n * sizeof(float);
    }
    size_t padded = n + ((GGUF_QBLK - n % GGUF_QBLK) % GGUF_QBLK);
    std::vector<float> buf(padded);
    for (size_t b = 0; b < padded; b += GGUF_QBLK) {
        float sc = 0;
        in.read((char*)&sc, 4);
        if (dtype == GGUF_DTYPE_Q8_0) {
            int8_t q[GGUF_QBLK];
            in.read((char*)q, GGUF_QBLK);
            decode_q80_block(sc, q, buf.data() + b);
        } else {
            uint8_t p[GGUF_QBLK / 2];
            in.read((char*)p, GGUF_QBLK / 2);
            decode_q40_block(sc, p, buf.data() + b);
        }
        if (in.fail()) return false;
    }
    for (size_t i = 0; i < n; ++i) t.data[i] = buf[i];
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
    auto params = model.parameters();  // const overload: vector<const Tensor*>
    size_t tensor_count = params.size();

    // KV metadata — config + general
    struct KV {
        std::string key, value;
    };
    std::vector<KV> kvs;
    // Use model num_parameters to infer nothing, but store config fields explicitly
    // We need config values: vocab_size, n_layers, n_heads, n_embd, block_size, weight_tying
    // Since GPT doesn't expose config directly, we deduce from parameters shapes:
    //  wte shape [vocab, n_embd] -> vocab, n_embd
    //  wpe shape [block_size, n_embd] -> block_size
    //  n_layers = (tensor_count - fixed) / per_block
    // Instead store via model.num_parameters? Better to store derived and also store config via
    // inspection. We will store what we can: retrieve via parameters but also need original config.
    // For now, store heuristic: we serialize config by reading first tensors.
    // To avoid hack, we store actual config fields by using a small trick: we have access via
    // private config_? Since we are not friend, we derive n_embd from wte shape[1], vocab from wte
    // shape[0], block_size from wpe shape[0]. n_layers we derive as blocks size: we can count
    // blocks via tensor_count pattern, but we store it directly as kvs. Alternative: we can store
    // all params' shapes as sufficient to reconstruct; but we also store textual config for
    // validation. We'll compute:
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

bool save_gguf_quant(const GPT& model, const std::string& path, int qtype) {
    uint32_t dtype = GGUF_DTYPE_F32;
    if (qtype == 8)
        dtype = GGUF_DTYPE_Q8_0;
    else if (qtype == 4)
        dtype = GGUF_DTYPE_Q4_0;
    {
        size_t slash = path.find_last_of("/\\");
        if (slash != std::string::npos) std::filesystem::create_directories(path.substr(0, slash));
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "[gguf] save_quant: cannot open " << path << "\n";
        return false;
    }
    auto params = model.parameters();
    size_t tensor_count = params.size();
    size_t vocab_size = tensor_count >= 1 ? params[0]->shape[0] : 0;
    size_t n_embd = tensor_count >= 1 ? params[0]->shape[1] : 0;
    size_t block_size = tensor_count >= 2 ? params[1]->shape[0] : 0;
    struct KV {
        std::string key, value;
    };
    std::vector<KV> kvs = {{"general.architecture", "llm"},
                           {"general.name", "llm-cpp"},
                           {"llm.vocab_size", std::to_string(vocab_size)},
                           {"llm.n_embd", std::to_string(n_embd)},
                           {"llm.block_size", std::to_string(block_size)},
                           {"llm.tensor_count", std::to_string(tensor_count)},
                           {"llm.quant_type", std::to_string(qtype)}};
    write_u32(out, GGUF_MAGIC);
    write_u32(out, GGUF_VERSION);
    write_u64(out, (uint64_t)tensor_count);
    write_u64(out, (uint64_t)kvs.size());
    for (auto& kv : kvs) {
        write_string(out, kv.key);
        write_u32(out, GGUF_TYPE_STRING);
        write_string(out, kv.value);
    }
    struct Info {
        std::string name;
        std::vector<uint64_t> dims;
        uint64_t offset = 0, nbytes = 0;
    };
    std::vector<Info> infos;
    for (size_t i = 0; i < tensor_count; ++i) {
        Info info;
        info.name = "tensor_" + std::to_string(i);
        for (auto d : params[i]->shape) info.dims.push_back((uint64_t)d);
        size_t padded =
            params[i]->data.size() + ((GGUF_QBLK - params[i]->data.size() % GGUF_QBLK) % GGUF_QBLK);
        info.nbytes = (dtype == GGUF_DTYPE_F32) ? params[i]->data.size() * sizeof(float)
                                                : gguf_nbytes(padded, dtype);
        infos.push_back(std::move(info));
    }
    size_t cur = 0;
    for (auto& info : infos) {
        cur = align_offset(cur, GGUF_ALIGN);
        info.offset = cur;
        cur += info.nbytes;
    }
    for (auto& info : infos) {
        write_string(out, info.name);
        write_u32(out, (uint32_t)info.dims.size());
        for (auto d : info.dims) write_u64(out, d);
        write_u32(out, dtype);
        write_u64(out, info.offset);
    }
    size_t hs = align_offset((size_t)out.tellp(), GGUF_ALIGN);
    for (size_t i = (size_t)out.tellp(); i < hs; ++i) out.put(0);
    size_t data_start = (size_t)out.tellp();
    for (size_t i = 0; i < infos.size(); ++i) {
        size_t target = data_start + infos[i].offset;
        for (size_t c = (size_t)out.tellp(); c < target; ++c) out.put(0);
        write_encoded(out, *params[i], dtype);
    }
    out.flush();
    std::cout << "[gguf] save_quant " << path << " qtype=" << qtype << " tensors " << tensor_count
              << "\n";
    return (bool)out;
}

bool load_gguf(GPT& model, const std::string& path) {
    try {
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
            std::cerr << "[gguf] load: unsupported version " << version << " expected "
                      << GGUF_VERSION << "\n";
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
        auto mutable_params = model.parameters();  // non-const for writing
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
            if (n_dims > 8) {  // H77: cap dims (model tensors are <= 2D)
                std::cerr << "[gguf] load: n_dims too large\n";
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
            if (info.dtype != GGUF_DTYPE_F32 && info.dtype != GGUF_DTYPE_Q8_0 &&
                info.dtype != GGUF_DTYPE_Q4_0) {
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
                    std::cerr << "[gguf] load: shape dim mismatch tensor " << info.name << " dim "
                              << d << " file " << info.dims[d] << " vs model " << p->shape[d]
                              << "\n";
                    return false;
                }
            }
            // I8 model weights are not GGUF-loadable (dequantize first); reject clearly
            if (p->is_int8()) {
                std::cerr << "[gguf] load: model tensor " << info.name << " is int8\n";
                return false;
            }
            if (info.offset + gguf_nbytes(p->data.size(), GGUF_DTYPE_F32) < info.offset &&
                info.dtype == GGUF_DTYPE_F32) {
                std::cerr << "[gguf] load: offset overflow\n";
                return false;
            }
        }

        // Now populate tensors (dequantizing Q8_0/Q4_0 on the fly)
        for (size_t i = 0; i < infos.size(); ++i) {
            const auto& info = infos[i];
            auto* p = mutable_params[i];
            size_t abs_offset = data_start + (size_t)info.offset;
            if (!read_decoded(in, *p, info.dtype, abs_offset)) {
                std::cerr << "[gguf] load: data read failed for " << info.name << "\n";
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
    } catch (const std::exception& e) {  // H77: corrupt files must reject, never throw out
        std::cerr << "[gguf] load: corrupt file (" << e.what() << ")\n";
        return false;
    }
}

}  // namespace llm
