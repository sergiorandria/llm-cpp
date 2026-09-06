#include "llm/alibi.h"

#include <cmath>
#include <vector>

namespace llm {

Tensor alibi_bias(size_t seq_len, size_t n_heads) {
    // Returns Tensor shape [n_heads, seq_len, seq_len] with ALiBi bias
    // slope per head: geometric sequence 2^{-8/n_heads * (h+1)}
    // bias[h][i][j] = -slope_h * (i - j) for j <= i, 0 otherwise? For causal we keep negative
    // For pure bias matrix without causal mask, bias = -slope * |i - j|
    Tensor out({n_heads, seq_len, seq_len}, 0.0f);
    // Compute slopes
    std::vector<float> slopes;
    slopes.reserve(n_heads);
    // ALiBi slopes: for n_heads power of 2, use 2^{-8/n * (h+1)}.
    // For non-power-of-2, paper uses interpolation; we approximate same formula.
    for (size_t h = 0; h < n_heads; ++h) {
        float slope = std::pow(2.0f, -8.0f * float(h + 1) / float(n_heads));
        slopes.push_back(slope);
    }
    for (size_t h = 0; h < n_heads; ++h) {
        float slope = slopes[h];
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                float dist = std::abs(float((int)i - (int)j));
                float bias = -slope * dist;
                // store at [h, i, j] flattened as h*seq*seq + i*seq + j
                size_t idx = h * seq_len * seq_len + i * seq_len + j;
                out.data[idx] = bias;
            }
        }
    }
    return out;
}

}  // namespace llm
