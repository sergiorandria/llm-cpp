#include "llm/eval.h"

#include <cmath>

#include "llm/loss.h"

namespace llm {

EvalResult evaluate(const GPT& model, const std::vector<std::vector<int>>& mmlu_data) {
    EvalResult res;
    res.mmlu = 0;
    res.hellaswag = 0;
    // Compute perplexity over provided data; if mmlu_data empty, use dummy
    // Perplexity = exp(avg cross-entropy)
    float total_loss = 0;
    size_t total_tokens = 0;
    if (mmlu_data.empty()) {
        res.ppl = 0;
        return res;
    }
    for (auto& seq : mmlu_data) {
        if (seq.empty()) continue;
        // Block size check: truncate if needed
        std::vector<int> batch = seq;
        // model forward
        auto logits = model.forward(batch);
        float loss = compute_loss(logits, batch);
        total_loss += loss * batch.size();
        total_tokens += batch.size();
    }
    if (total_tokens > 0) {
        float avg_loss = total_loss / float(total_tokens);
        res.ppl = std::exp(avg_loss);
    } else {
        res.ppl = 0;
    }
    // mmlu/hellaswag datasets not available in this repo — stub to 0 with comment
    // Real evaluation would require external datasets; we report 0 to indicate not evaluated
    res.mmlu = 0;
    res.hellaswag = 0;
    return res;
}

}  // namespace llm
