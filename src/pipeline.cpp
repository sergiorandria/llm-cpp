#include "llm/pipeline.h"

#include "llm/loss.h"

namespace llm {

Pipeline::Pipeline(GPT& model, size_t stages, size_t micro_batch)
    : model_(model), stages_(stages), micro_(micro_batch) {}

void Pipeline::train_step(const std::vector<int>& batch) {
    if (batch.empty()) return;
    // Pipeline parallelism stub: split batch into micro-batches and accumulate grads sequentially
    // Since we have no real multi-device, we simulate pipeline by sequential micro-batch processing
    // Equivalent to gradient accumulation with micro_ sized chunks
    model_.zero_grad();
    size_t num_micro = (batch.size() + micro_ - 1) / micro_;
    // Store accumulated grads? We accumulate directly into model grad fields via backward calls
    for (size_t m = 0; m < num_micro; ++m) {
        size_t start = m * micro_;
        size_t end = std::min(start + micro_, batch.size());
        std::vector<int> micro_batch(batch.begin() + start, batch.begin() + end);
        // Forward + compute loss grad for this micro batch
        auto [logits, hidden] = model_.forward_with_hidden(micro_batch);
        size_t T = logits.shape[0], V = logits.shape[1];
        Tensor dlogits = logits.softmax(1);
        for (size_t i = 0; i < T && i < micro_batch.size(); ++i) {
            int tgt = micro_batch[i];
            tgt = ((tgt % (int)V) + (int)V) % (int)V;
            dlogits(i, tgt) -= 1.0f;
            for (size_t j = 0; j < V; ++j) dlogits(i, j) /= float(T);
        }
        // Scale grad by micro_batch proportion to keep overall batch mean
        float scale = float(micro_batch.size()) / float(batch.size());
        for (auto& v : dlogits.data) v *= scale;
        // Backward accumulates grads (we don't zero between micros)
        // Need to avoid double-counting wte/wpe? backward accumulates, so fine
        // But model.backward expects hidden from same forward; we pass it
        model_.backward(dlogits, micro_batch, hidden);
        // Note: stages_ not used for real pipeline partition; we simulate overlap by sequential
        (void)stages_;
    }
    // Caller should do optimizer step after accumulating
    // For standalone Pipeline::train_step we do a small SGD update to make change observable
    float lr = 1e-3f;
    for (auto* p : model_.parameters()) {
        for (size_t i = 0; i < p->data.size() && i < p->grad.size(); ++i)
            p->data[i] -= lr * p->grad[i];
    }
}

}  // namespace llm
