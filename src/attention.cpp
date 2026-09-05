#include "llm/attention.h"
#include "llm/flash_attention.h"
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
    // x: [T, C] where C = n_heads * head_dim; true per-head attention
    assert(n_embd_ % n_heads_ == 0);
    size_t T = x.shape[0];
    Tensor Q = x.matmul(Wq_); // [T, C]
    Tensor K = x.matmul(Wk_);
    Tensor V = x.matmul(Wv_);

    auto slice_head = [&](const Tensor& t, size_t h) -> Tensor {
        Tensor out({T, head_dim_}, 0.0f);
        size_t base = h * head_dim_;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < head_dim_; ++j)
                out(i, j) = t(i, base + j);
        return out;
    };

    Tensor out({T, n_embd_}, 0.0f);
    float scale = 1.0f / std::sqrt((float)head_dim_);

    for (size_t h = 0; h < n_heads_; ++h) {
        Tensor Qh = slice_head(Q, h);
        Tensor Kh = slice_head(K, h);
        Tensor Vh = slice_head(V, h);
        Tensor out_h;
        if (T > 128) {
            // FlashAttention tiled path: O(n) memory, cache-friendly
            out_h = flash_attention(Qh, Kh, Vh, scale, causal);
            if (dropout_p > 0.0f) {
                std::mt19937 rng(123 + h);
                out_h = out_h.dropout(dropout_p, rng);
            }
        } else {
            Tensor Kt = Kh.transpose();
            Tensor scores = Qh.matmul(Kt);
            for (auto& v : scores.data) v *= scale;
            if (causal) {
                for (size_t i = 0; i < T; ++i)
                    for (size_t j = i + 1; j < T; ++j)
                        scores(i, j) = -1e9f;
            }
            Tensor attn = scores.softmax(1);
            if (dropout_p > 0.0f) {
                std::mt19937 rng(123 + h);
                attn = attn.dropout(dropout_p, rng);
            }
            out_h = attn.matmul(Vh);
        }
        // concat back
        size_t base = h * head_dim_;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < head_dim_; ++j)
                out(i, base + j) = out_h(i, j);
    }

    Tensor proj = out.matmul(Wo_); // [T, C]
    return proj;
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
    return {&Wq_, &Wk_, &Wv_, &Wo_, &bq_, &bk_, &bv_, &bo_};
}
std::vector<const Tensor*> MultiHeadAttention::parameters() const {
    return {&Wq_, &Wk_, &Wv_, &Wo_, &bq_, &bk_, &bv_, &bo_};
}

} // namespace llm
