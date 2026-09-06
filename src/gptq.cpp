#include "llm/gptq.h"

#include <algorithm>
#include <cmath>

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

}  // namespace llm
