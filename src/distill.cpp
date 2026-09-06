#include "llm/distill.h"

#include <cmath>

#include "llm/loss.h"

namespace llm {

void distill_step(GPT& student, const GPT& teacher, const std::vector<int>& batch, float temp) {
    // Distillation via KL divergence between teacher and student softened logits
    // Requires real gradients (student must have backward)
    if (batch.empty()) return;
    student.zero_grad();
    // Teacher forward (no grad) with temperature
    auto teacher_logits = teacher.forward(batch);
    auto [student_logits, hidden] = student.forward_with_hidden(batch);
    size_t T = student_logits.shape[0], V = student_logits.shape[1];
    // Soften with temperature
    Tensor t_probs = teacher_logits;
    Tensor s_logits = student_logits;
    for (auto& v : t_probs.data) v /= temp;
    for (auto& v : s_logits.data) v /= temp;
    Tensor p_t = t_probs.softmax(1);   // teacher probs
    Tensor p_s = s_logits.softmax(1);  // student probs (logits/temp)
    // KL divergence gradient w.r.t s_logits: dL/dz_s = (p_s - p_t) / T * temp? Actually loss =
    // temp^2 * KL, so grad ~ temp * (p_s - p_t)/T Compute dlogits = (p_s - p_t) * (temp / T) ???
    // For temp scaling, typical factor temp^2
    Tensor dlogits({T, V}, 0.0f);
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < V; ++j) {
            float diff = p_s(i, j) - p_t(i, j);
            dlogits(i, j) = diff * (temp * temp) / float(T) / temp;  // = temp*(p_s-p_t)/T
            // Simplify to temp*(p_s-p_t)/T
        }
    }
    // Adjust: if we used temp^2 * KL, grad = temp*(p_s - p_t)/T? We'll just use (p_s - p_t)/T *
    // temp
    student.backward(dlogits, batch, hidden);
    // Note: caller is responsible for optimizer step; this just accumulates grads
}

}  // namespace llm
