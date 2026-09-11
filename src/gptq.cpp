#include "llm/gptq.h"
#include "llm/checkpoint.h"
#include "llm/model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>

namespace llm {

Tensor quantize_4bit(const Tensor& x, size_t group) {
    Tensor scale_dummy;
    return quantize_4bit(x, scale_dummy, group);
}

Tensor quantize_4bit(const Tensor& x, Tensor& scale_out, size_t group) {
    // Group-wise 4-bit quantization: each group of `group` elements has its own scale
    // Returns quantized int4 stored as float in range [-8,7], shape same as x
    // scale_out will be filled with per-group scales shape [num_groups]
    Tensor q(x.shape, 0.0f);
    size_t n = x.data.size();
    size_t num_groups = (n + group - 1) / group;
    scale_out = Tensor({num_groups}, 0.0f);
    for (size_t g = 0; g < n; g += group) {
        size_t g_idx = g / group;
        size_t g_end = std::min(g + group, n);
        float max_abs = 0;
        for (size_t i = g; i < g_end; ++i) max_abs = std::max(max_abs, std::abs(x.data[i]));
        float scale = max_abs / 7.0f + 1e-8f;  // 4bit signed 3 bits magnitude + sign => 7 levels
        scale_out.data[g_idx] = scale;
        for (size_t i = g; i < g_end; ++i) {
            float v = std::round(x.data[i] / scale);
            v = std::max(-8.0f, std::min(7.0f, v));
            q.data[i] = v;
        }
    }
    return q;
}

Tensor dequantize_4bit(const Tensor& q, const Tensor& scale, size_t group) {
    // q: quantized tensor [same shape as orig]
    // scale: per-group scale tensor shape [num_groups]
    Tensor out(q.shape, 0.0f);
    size_t n = q.data.size();
    size_t num_groups = (n + group - 1) / group;
    assert(scale.data.size() >= num_groups || scale.data.size() == num_groups ||
           scale.data.size() == 1);
    for (size_t g = 0; g < num_groups; ++g) {
        float s = scale.data[g % scale.data.size()];
        size_t start = g * group;
        size_t end = std::min(start + group, n);
        for (size_t i = start; i < end; ++i) out.data[i] = q.data[i] * s;
    }
    return out;
}

// ── F52: model-level 4-bit + persistence ──
std::vector<GPTQEntry> quantize_model_4bit(GPT& model, size_t group) {
    std::vector<GPTQEntry> entries;
    auto named = gpt_named_params(model);
    for (auto& kv : named) {
        if (kv.second->shape.size() != 2) continue;  // quantize 2D weights only
        GPTQEntry e;
        e.name = kv.first;
        e.group = group;
        e.q = quantize_4bit(*kv.second, e.scale, group);
        entries.push_back(std::move(e));
    }
    return entries;
}

void save_gptq(const std::string& path, const std::vector<GPTQEntry>& entries) {
    std::ofstream out(path, std::ios::binary);
    uint32_t magic = 0x47505451;  // "GPTQ"
    uint32_t ver = 1;
    out.write((char*)&magic, 4);
    out.write((char*)&ver, 4);
    uint64_t n = entries.size();
    out.write((char*)&n, 8);
    auto write_t = [&](const Tensor& t) {
        uint64_t nd = t.shape.size();
        out.write((char*)&nd, 8);
        for (auto d : t.shape) { uint64_t v = d; out.write((char*)&v, 8); }
        uint64_t m = t.data.size();
        out.write((char*)&m, 8);
        out.write((char*)t.data.data(), m * sizeof(float));
    };
    for (auto& e : entries) {
        uint64_t nl = e.name.size(), g = e.group;
        out.write((char*)&nl, 8);
        out.write(e.name.data(), nl);
        out.write((char*)&g, 8);
        write_t(e.q);
        write_t(e.scale);
    }
}

std::vector<GPTQEntry> load_gptq(const std::string& path) {
    std::vector<GPTQEntry> entries;
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "[gptq] cannot open " << path << "\n"; return entries; }
    uint32_t magic = 0, ver = 0;
    in.read((char*)&magic, 4);
    in.read((char*)&ver, 4);
    if (magic != 0x47505451) { std::cerr << "[gptq] bad magic\n"; return entries; }
    uint64_t n = 0;
    in.read((char*)&n, 8);
    auto read_t = [&](Tensor& t) {
        uint64_t nd = 0;
        in.read((char*)&nd, 8);
        std::vector<size_t> shape(nd);
        for (uint64_t i = 0; i < nd; ++i) { uint64_t v = 0; in.read((char*)&v, 8); shape[i] = v; }
        uint64_t m = 0;
        in.read((char*)&m, 8);
        t = Tensor(shape, 0.0f);
        in.read((char*)t.data.data(), m * sizeof(float));
    };
    for (uint64_t i = 0; i < n; ++i) {
        GPTQEntry e;
        uint64_t nl = 0, g = 128;
        in.read((char*)&nl, 8);
        e.name.assign(nl, '\0');
        in.read(e.name.data(), nl);
        in.read((char*)&g, 8);
        e.group = (size_t)g;
        read_t(e.q);
        read_t(e.scale);
        entries.push_back(std::move(e));
    }
    return entries;
}

// ── F53: AWQ channel rescale ──
Tensor channel_act_mag(const Tensor& activations) {
    // activations [T, C] -> mag [C] mean abs per channel
    size_t T = activations.shape[0], C = activations.shape[1];
    Tensor mag({C}, 0.0f);
    for (size_t j = 0; j < C; ++j) {
        double s = 0;
        for (size_t i = 0; i < T; ++i) s += std::abs(activations(i, j));
        mag.data[j] = (float)(s / T);
    }
    return mag;
}

std::pair<Tensor, Tensor> awq_rescale_for_quant(const Tensor& weight, const Tensor& act_mag, float alpha) {
    // weight [in, out] scaled per input channel: W'[:,j] rows? We scale per ROW (in-channel):
    // s_i = (mag_i / max)^alpha; W'[i,j] = W[i,j]*s_i; inv[i] = 1/s_i.
    assert(weight.shape.size() == 2 && act_mag.data.size() == weight.shape[0]);
    float mx = 0;
    for (auto v : act_mag.data) mx = std::max(mx, v);
    if (mx <= 0) mx = 1.0f;
    Tensor scaled = weight, inv({weight.shape[0]}, 1.0f);
    for (size_t i = 0; i < weight.shape[0]; ++i) {
        float s = std::pow(act_mag.data[i] / mx, alpha);
        if (s < 1e-3f) s = 1e-3f;
        inv.data[i] = 1.0f / s;
        for (size_t j = 0; j < weight.shape[1]; ++j) scaled(i, j) = weight(i, j) * s;
    }
    return {scaled, inv};
}

}  // namespace llm
