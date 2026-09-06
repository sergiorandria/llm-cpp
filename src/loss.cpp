#include "llm/loss.h"

#include <algorithm>
#include <cmath>
namespace llm {
float cross_entropy_loss(const Tensor& logits, const std::vector<int>& targets) {
    float loss = 0;
    size_t T = logits.shape[0];
    for (size_t i = 0; i < T && i < targets.size(); ++i) {
        float maxv = logits(i, 0);
        for (size_t j = 1; j < logits.shape[1]; ++j) maxv = std::max(maxv, logits(i, j));
        float sum = 0;
        for (size_t j = 0; j < logits.shape[1]; ++j) sum += std::exp(logits(i, j) - maxv);
        int tgt = targets[i];
        // Clamp OOV target to vocab range to avoid OOB
        int vocab = (int)logits.shape[1];
        tgt = ((tgt % vocab) + vocab) % vocab;
        float logp = logits(i, tgt) - maxv - std::log(sum);
        loss -= logp;
    }
    return loss / T;
}
float compute_loss(const Tensor& logits, const std::vector<int>& targets) {
    return cross_entropy_loss(logits, targets);
}
}  // namespace llm
