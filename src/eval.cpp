#include "llm/eval.h"

#include <cmath>

#include "llm/loss.h"

namespace llm {

EvalResult evaluate(const GPT& model, const std::vector<std::vector<int>>& data) {
    EvalResult res;
    res.mmlu = 0;
    res.hellaswag = 0;
    res.ppl = 0;
    if (data.empty()) {
        return res;
    }
    float total_loss = 0;
    size_t total_tokens = 0;
    size_t correct = 0;
    size_t total_pred = 0;
    for (auto& seq : data) {
        if (seq.empty()) continue;
        // Truncate to block_size if needed
        std::vector<int> batch = seq;
        const auto& cfg = model.config();
        if(batch.size() > cfg.block_size){
            batch.assign(seq.end() - cfg.block_size, seq.end());
        }
        auto logits = model.forward(batch);
        float loss = compute_loss(logits, batch);
        total_loss += loss * batch.size();
        total_tokens += batch.size();
        // Token accuracy for mmlu proxy: predict next token via argmax
        for(size_t i=0;i+1<batch.size() && i+1<logits.shape[0];++i){
            size_t pred = logits.argmax(i);
            int target = batch[i+1];
            // Clamp target to vocab
            int vocab = (int)cfg.vocab_size;
            target = ((target % vocab) + vocab) % vocab;
            if((int)pred == target) correct++;
            total_pred++;
        }
    }
    if (total_tokens > 0) {
        float avg_loss = total_loss / float(total_tokens);
        res.ppl = std::exp(avg_loss);
    }
    if(total_pred>0){
        res.mmlu = float(correct) / float(total_pred);
        // HellaSwag proxy: same as mmlu for now, but could be per-sequence accuracy
        // For true HellaSwag, would compare log-likelihood of choices; we approximate with same accuracy
        res.hellaswag = res.mmlu;
    }
    return res;
}

}  // namespace llm
