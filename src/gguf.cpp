#include "llm/gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
static constexpr uint32_t GGUF_DTYPE_Q4_K = 12;  // ggml type id, 256-elem super-blocks (144B)
static constexpr size_t GGUF_QBLK = 32;
static constexpr size_t GGUF_Q4K_BLK = 256;
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

// ── Q4_K super-block codec ──
static uint16_t f32_to_f16(float v) {
    uint32_t b = 0;
    memcpy(&b, &v, sizeof(b));
    uint32_t sign = (b >> 16) & 0x8000u;
    int exp = int((b >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = b & 0x7FFFFFu;
    if (exp >= 31) return (uint16_t)(sign | 0x7BFFu);  // inf/nan -> inf
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;  // underflow to signed zero
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(1 - exp);
        uint32_t half = mant >> shift;
        uint32_t rest = mant & ((shift >= 32) ? 0xFFFFFFFFu : ((1u << shift) - 1u));
        uint32_t halfway = 1u << (shift - 1);
        if (rest > halfway || (rest == halfway && (half & 1u))) ++half;
        return (uint16_t)(sign | (half & 0x3FFu));
    }
    uint32_t half = mant >> 13;
    uint32_t rest = mant & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) {
        ++half;
        if (half == 0x400u) {
            half = 0;
            if (++exp >= 31) return (uint16_t)(sign | 0x7BFFu);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | half);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    int exp = ((h >> 10) & 0x1Fu);
    uint32_t mant = (uint32_t)(h & 0x3FFu);
    uint32_t b;
    if (exp == 0) {
        if (mant == 0) {
            b = sign;  // signed zero
        } else {
            // subnormal: normalize
            exp = 1;
            while (!(mant & 0x400u)) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFu;
            b = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        b = sign | 0x7F800000u | (mant << 13);  // inf/nan
    } else {
        b = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    }
    float v = 0;
    memcpy(&v, &b, sizeof(v));
    return v;
}

// Unpack one (sc, mn) 6-bit pair for sub-block j (ggml get_scale_min_k4 order)
static void q4k_unpack_scales(const uint8_t* scales, int j, uint8_t& sc, uint8_t& mn) {
    if (j < 4) {
        sc = scales[j] & 63;
        mn = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

void encode_q4k_block(const float* x, uint8_t* out144) {
    // out144 layout: d f16 [0:2], dmin f16 [2:4], scales[12] [4:16], qs[128] [16:144]
    float mn[8], mx[8];
    // Non-finite super-block -> emit zeros (documented; never silently NaN)
    bool finite = true;
    for (int i = 0; i < 256; ++i)
        if (!std::isfinite(x[i])) {
            finite = false;
            break;
        }
    for (int j = 0; j < 8; ++j) {
        mn[j] = finite ? x[32 * j] : 0.0f;
        mx[j] = finite ? x[32 * j] : 0.0f;
        for (int l = 1; l < 32 && finite; ++l) {
            float v = x[32 * j + l];
            if (v < mn[j]) mn[j] = v;
            if (v > mx[j]) mx[j] = v;
        }
    }
    float sc_true[8];
    float dall_anchor = 0;
    if (finite) {
        for (int j = 0; j < 8; ++j) {
            sc_true[j] = (mx[j] - mn[j]) / 15.0f;
            dall_anchor = std::max(dall_anchor, sc_true[j]);
        }
    } else {
        for (int j = 0; j < 8; ++j) {
            mn[j] = 0;
            sc_true[j] = 0;
        }
    }
    float dall = dall_anchor / 63.0f;
    // Min anchor: max(-mn) when some sub-min is negative, else -max(mn)
    // (exact for single-sign-min blocks; see header docs).
    float neg_max = 0;  // max(-mn) over j with mn<0, else 0
    float pos_max = 0;  // max(mn) over j with mn>=0
    for (int j = 0; j < 8; ++j) {
        if (mn[j] < 0)
            neg_max = std::max(neg_max, -mn[j]);
        else
            pos_max = std::max(pos_max, mn[j]);
    }
    float dmin_anchor = (neg_max > 0) ? neg_max : -pos_max;  // 0 iff all mn==0
    float dmin = dmin_anchor / 63.0f;
    uint8_t qsc[8], qmn[8];
    for (int j = 0; j < 8; ++j) {
        qsc[j] = (dall_anchor > 0) ? (uint8_t)std::max(0.0f, std::min(63.0f,
                                          std::round(63.0f * sc_true[j] / dall_anchor)))
                                   : 0;
        qmn[j] = (dmin_anchor != 0) ? (uint8_t)std::max(0.0f, std::min(63.0f,
                                           std::round(63.0f * (-mn[j]) / dmin_anchor)))
                                    : 0;
    }
    out144[0] = (uint8_t)(f32_to_f16(dall) & 0xFF);
    out144[1] = (uint8_t)(f32_to_f16(dall) >> 8);
    out144[2] = (uint8_t)(f32_to_f16(dmin) & 0xFF);
    out144[3] = (uint8_t)(f32_to_f16(dmin) >> 8);
    uint8_t* scales = out144 + 4;
    for (int j = 0; j < 4; ++j) {
        scales[j] = (qsc[j] & 63) | ((qsc[j + 4] >> 4) << 6);
        scales[j + 4] = (qmn[j] & 63) | ((qmn[j + 4] >> 4) << 6);
        scales[8 + j] = (qsc[j + 4] & 0xF) | ((qmn[j + 4] & 0xF) << 4);
    }
    // Quantize values against the QUANTIZED scales/mins (what dequant uses)
    float dall_f = f16_to_f32(f32_to_f16(dall));
    float dmin_f = f16_to_f32(f32_to_f16(dmin));
    uint8_t* qs = out144 + 16;
    for (int i = 0; i < 128; ++i) qs[i] = 0;
    for (int i = 0; i < 256; ++i) {
        int j = i / 32;
        float dj = dall_f * (float)qsc[j];
        float dmj = dmin_f * (float)qmn[j];
        int q = 0;
        if (dj != 0) {
            q = (int)std::round((x[i] + dmj) / dj);
            q = std::max(0, std::min(15, q));
        }
        int il = i / 64, in = i % 64;
        int b = 32 * il + (in & 31);
        if (in < 32)
            qs[b] |= (uint8_t)q;
        else
            qs[b] |= (uint8_t)(q << 4);
    }
}

void decode_q4k_block(const uint8_t* in144, float* out) {
    uint16_t dh = (uint16_t)in144[0] | ((uint16_t)in144[1] << 8);
    uint16_t mh = (uint16_t)in144[2] | ((uint16_t)in144[3] << 8);
    float dall = f16_to_f32(dh);
    float dmin = f16_to_f32(mh);
    const uint8_t* scales = in144 + 4;
    const uint8_t* qs = in144 + 16;
    for (int i = 0; i < 256; ++i) {
        int il = i / 64, in = i % 64;
        int j = 2 * il + (in >= 32 ? 1 : 0);
        uint8_t sc, mn;
        q4k_unpack_scales(scales, j, sc, mn);
        int b = 32 * il + (in & 31);
        int q = (in < 32) ? (qs[b] & 0xF) : (qs[b] >> 4);
        out[i] = dall * (float)sc * (float)q - dmin * (float)mn;
    }
}

// Encoded size of a tensor with n floats under dtype tag (n pre-padded)
static size_t gguf_nbytes(size_t n, uint32_t dtype) {
    if (dtype == GGUF_DTYPE_Q8_0) return (n / GGUF_QBLK) * (4 + GGUF_QBLK);
    if (dtype == GGUF_DTYPE_Q4_0) return (n / GGUF_QBLK) * (4 + GGUF_QBLK / 2);
    if (dtype == GGUF_DTYPE_Q4_K) return (n / GGUF_Q4K_BLK) * 144;
    return n * sizeof(float);
}

// Pad length for a dtype's block size
static size_t gguf_pad_len(size_t n, uint32_t dtype) {
    size_t blk = (dtype == GGUF_DTYPE_Q4_K) ? GGUF_Q4K_BLK : GGUF_QBLK;
    if (dtype == GGUF_DTYPE_F32) return n;
    return n + ((blk - n % blk) % blk);
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
    } else if (dtype == GGUF_DTYPE_Q4_K) {
        std::vector<float> padded4k(t.data.begin(), t.data.end());
        while (padded4k.size() % GGUF_Q4K_BLK) padded4k.push_back(0.0f);
        for (size_t b = 0; b < padded4k.size(); b += GGUF_Q4K_BLK) {
            uint8_t blk[144];
            encode_q4k_block(padded4k.data() + b, blk);
            out.write((char*)blk, 144);
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
    if (dtype == GGUF_DTYPE_Q4_K) {
        size_t padded = gguf_pad_len(n, dtype);
        std::vector<float> buf(padded);
        for (size_t b = 0; b < padded; b += GGUF_Q4K_BLK) {
            uint8_t blk[144];
            in.read((char*)blk, 144);
            if (in.fail()) return false;
            decode_q4k_block(blk, buf.data() + b);
        }
        for (size_t i = 0; i < n; ++i) t.data[i] = buf[i];
        return true;
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
    else if (qtype == 12)
        dtype = GGUF_DTYPE_Q4_K;
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
        size_t padded = gguf_pad_len(params[i]->data.size(), dtype);
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
                info.dtype != GGUF_DTYPE_Q4_0 && info.dtype != GGUF_DTYPE_Q4_K) {
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
