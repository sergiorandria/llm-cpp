#include "llm/mqa.h"

#include <cmath>

namespace llm {

Tensor mqa_forward(const Tensor& x, const Tensor& Wq, const Tensor& Wkv, size_t n_heads,
                   size_t head_dim) {
    // x: [T, n_embd]
    // Wq: [n_embd, n_heads*head_dim]
    // Wkv: [n_embd, 2*head_dim] -> single K and V shared
    // Returns: [T, n_heads*head_dim]
    size_t T = x.shape[0];
    size_t n_embd = x.shape[1];
    (void)n_embd;
    assert(Wq.shape[0] == x.shape[1]);
    assert(Wkv.shape[0] == x.shape[1]);
    assert(Wq.shape[1] == n_heads * head_dim);
    assert(Wkv.shape[1] == 2 * head_dim);
    Tensor Q = x.matmul(Wq);    // [T, n_heads*hd]
    Tensor KV = x.matmul(Wkv);  // [T, 2*hd]
    // Split K and V
    Tensor K({T, head_dim}, 0.0f), V({T, head_dim}, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < head_dim; ++d) {
            K(t, d) = KV(t, d);
            V(t, d) = KV(t, head_dim + d);
        }
    }
    float scale = 1.0f / std::sqrt(float(head_dim));
    Tensor out({T, n_heads * head_dim}, 0.0f);
    // For each head, compute attention with shared K,V
    Tensor Kt = K.transpose();  // [hd, T]
    for (size_t h = 0; h < n_heads; ++h) {
        // Slice Qh
        Tensor Qh({T, head_dim}, 0.0f);
        size_t base = h * head_dim;
        for (size_t t = 0; t < T; ++t)
            for (size_t d = 0; d < head_dim; ++d) Qh(t, d) = Q(t, base + d);
        Tensor scores = Qh.matmul(Kt);  // [T, T]
        for (auto& v : scores.data) v *= scale;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = i + 1; j < T; ++j) scores(i, j) = -1e9f;
        Tensor attn = scores.softmax(1);
        Tensor out_h = attn.matmul(V);  // [T, hd]
        for (size_t t = 0; t < T; ++t)
            for (size_t d = 0; d < head_dim; ++d) out(t, base + d) = out_h(t, d);
    }
    return out;
}

Tensor gqa_forward(const Tensor& x, const Tensor& Wq, const Tensor& Wk, const Tensor& Wv,
                   size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    assert(n_heads % n_kv_heads == 0);
    assert(Wq.shape[1] == n_heads * head_dim);
    assert(Wk.shape[1] == n_kv_heads * head_dim);
    assert(Wv.shape[1] == n_kv_heads * head_dim);
    size_t T = x.shape[0];
    Tensor Q = x.matmul(Wq);  // [T, nq*hd]
    Tensor K = x.matmul(Wk);  // [T, nkv*hd]
    Tensor V = x.matmul(Wv);
    float scale = 1.0f / std::sqrt(float(head_dim));
    size_t group = n_heads / n_kv_heads;
    Tensor out({T, n_heads * head_dim}, 0.0f);
    for (size_t h = 0; h < n_heads; ++h) {
        size_t kv = h / group;
        Tensor Qh({T, head_dim}, 0.0f), Kh({T, head_dim}, 0.0f), Vh({T, head_dim}, 0.0f);
        for (size_t t = 0; t < T; ++t)
            for (size_t d = 0; d < head_dim; ++d) {
                Qh(t, d) = Q(t, h * head_dim + d);
                Kh(t, d) = K(t, kv * head_dim + d);
                Vh(t, d) = V(t, kv * head_dim + d);
            }
        Tensor Kt = Kh.transpose();
        Tensor scores = Qh.matmul(Kt);
        for (auto& v : scores.data) v *= scale;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = i + 1; j < T; ++j) scores(i, j) = -1e9f;
        Tensor attn = scores.softmax(1);
        Tensor out_h = attn.matmul(Vh);
        for (size_t t = 0; t < T; ++t)
            for (size_t d = 0; d < head_dim; ++d) out(t, h * head_dim + d) = out_h(t, d);
    }
    return out;
}

}  // namespace llm
