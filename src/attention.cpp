#include "llm/attention.h"
#include <cmath>

namespace llm {

MultiHeadAttention::MultiHeadAttention(size_t n_embd, size_t n_heads, size_t block_size, bool bias)
    : n_heads_(n_heads), n_embd_(n_embd), head_dim_(n_embd / n_heads),
      Wq_({n_embd, n_embd}), Wk_({n_embd, n_embd}), Wv_({n_embd, n_embd}), Wo_({n_embd, n_embd}),
      bq_({n_embd}), bk_({n_embd}), bv_({n_embd}), bo_({n_embd}) {
    (void)block_size; (void)bias;
    Wq_.randn(0, 0.02f); Wk_.randn(0, 0.02f); Wv_.randn(0, 0.02f); Wo_.randn(0, 0.02f);
}

Tensor MultiHeadAttention::forward(const Tensor& x, bool causal, float dropout_p) const {
    // x: [T, C]
    size_t T = x.shape[0];
    // Naive: Q = x @ Wq, etc.
    Tensor Q = x.matmul(Wq_);
    Tensor K = x.matmul(Wk_);
    Tensor V = x.matmul(Wv_);

    // Scaled dot-product attention (single head for skeleton; split heads properly in full impl)
    // scores = Q @ K^T / sqrt(head_dim)
    Tensor Kt = K.transpose();
    Tensor scores = Q.matmul(Kt); // [T, T]
    float scale = 1.0f / std::sqrt((float)head_dim_);
    for (auto& v : scores.data) v *= scale;

    // Causal mask
    if(causal){
        for (size_t i = 0; i < T; ++i) {
            for (size_t j = i + 1; j < T; ++j) {
                scores(i, j) = -1e9f;
            }
        }
    }
    (void)dropout_p; // TODO: apply dropout on attn

    Tensor attn = scores.softmax(1); // [T, T]
    Tensor out = attn.matmul(V);     // [T, C]
    Tensor proj = out.matmul(Wo_);   // [T, C]
    return proj;
}

} // namespace llm
