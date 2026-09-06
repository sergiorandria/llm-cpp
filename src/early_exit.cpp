#include "llm/early_exit.h"

#include <algorithm>
#include <cmath>

namespace llm {

bool should_early_exit(const Tensor& logits, float thresh) {
    // logits: [T, vocab] or [vocab] or [T, C]
    // Take last row softmax, compute max prob, compare to thresh
    if (logits.data.empty()) return false;
    size_t V = logits.shape.empty() ? logits.data.size() : logits.shape.back();
    size_t T = logits.shape.size() >= 2 ? logits.shape[0] : 1;
    // Get last token logits
    size_t row = T - 1;
    // Compute max for softmax stability
    float maxv = logits.data[row * V];
    for (size_t j = 1; j < V; ++j) maxv = std::max(maxv, logits.data[row * V + j]);
    float sum = 0;
    float maxp = 0;
    for (size_t j = 0; j < V; ++j) {
        float p = std::exp(logits.data[row * V + j] - maxv);
        sum += p;
    }
    for (size_t j = 0; j < V; ++j) {
        float p = std::exp(logits.data[row * V + j] - maxv) / sum;
        maxp = std::max(maxp, p);
    }
    return maxp > thresh;
}

}  // namespace llm
