#include "llm/distill.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "llm/loss.h"

namespace llm {

static Tensor kl_dlogits(const Tensor& teacher_logits, const Tensor& student_logits, float temp) {
    size_t T = student_logits.shape[0], V = student_logits.shape[1];
    Tensor t_probs = teacher_logits;
    Tensor s_logits = student_logits;
    for (auto& v : t_probs.data) v /= temp;
    for (auto& v : s_logits.data) v /= temp;
    Tensor p_t = t_probs.softmax(1);
    Tensor p_s = s_logits.softmax(1);
    Tensor dlogits({T, V}, 0.0f);
    for (size_t i = 0; i < T; ++i)
        for (size_t j = 0; j < V; ++j) dlogits(i, j) = (p_s(i, j) - p_t(i, j)) * temp / float(T);
    return dlogits;
}

void distill_step(GPT& student, const GPT& teacher, const std::vector<int>& batch, float temp) {
    // Distillation via KL divergence between teacher and student softened logits
    // Requires real gradients (student must have backward)
    if (batch.empty()) return;
    student.zero_grad();
    // Teacher forward (no grad) with temperature
    auto teacher_logits = teacher.forward(batch);
    auto [student_logits, hidden] = student.forward_with_hidden(batch);
    Tensor dlogits = kl_dlogits(teacher_logits, student_logits, temp);
    student.backward(dlogits, batch, hidden);
    // Note: caller is responsible for optimizer step; this just accumulates grads
}

void cache_teacher_logits(const GPT& teacher, const std::vector<std::vector<int>>& batches,
                          const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    uint64_t n = batches.size();
    out.write((char*)&n, 8);
    for (auto& b : batches) {
        Tensor l = teacher.forward(b);
        uint64_t T = l.shape[0], V = l.shape[1];
        out.write((char*)&T, 8);
        out.write((char*)&V, 8);
        out.write((char*)l.data.data(), l.data.size() * sizeof(float));
    }
}

std::vector<Tensor> load_cached_logits(const std::string& path) {
    std::vector<Tensor> out;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[distill] cannot open cache " << path << "\n";
        return out;
    }
    uint64_t n = 0;
    in.read((char*)&n, 8);
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t T = 0, V = 0;
        in.read((char*)&T, 8);
        in.read((char*)&V, 8);
        Tensor l({(size_t)T, (size_t)V}, 0.0f);
        in.read((char*)l.data.data(), l.data.size() * sizeof(float));
        out.push_back(std::move(l));
    }
    return out;
}

void distill_step_cached(GPT& student, const Tensor& teacher_logits, const std::vector<int>& batch,
                         float temp) {
    if (batch.empty()) return;
    student.zero_grad();
    auto [student_logits, hidden] = student.forward_with_hidden(batch);
    Tensor dlogits = kl_dlogits(teacher_logits, student_logits, temp);
    student.backward(dlogits, batch, hidden);
}

}  // namespace llm
