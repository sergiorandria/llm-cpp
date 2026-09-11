#include "llm/attention.h"

#include <cmath>

#include "llm/flash_attention.h"
#include "llm/flash_config.h"
#include "llm/profiling.h"

namespace llm {

MultiHeadAttention::MultiHeadAttention(size_t n_embd, size_t n_heads, size_t block_size, bool bias)
    : n_heads_(n_heads),
      n_embd_(n_embd),
      head_dim_(n_embd / n_heads),
      bias_(bias),
      Wq_({n_embd, n_embd}),
      Wk_({n_embd, n_embd}),
      Wv_({n_embd, n_embd}),
      Wo_({n_embd, n_embd}),
      bq_({n_embd}),
      bk_({n_embd}),
      bv_({n_embd}),
      bo_({n_embd}) {
    (void)block_size;
    Wq_.randn(0, 0.02f);
    Wk_.randn(0, 0.02f);
    Wv_.randn(0, 0.02f);
    Wo_.randn(0, 0.02f);
    bq_.fill(0.0f);
    bk_.fill(0.0f);
    bv_.fill(0.0f);
    bo_.fill(0.0f);
}

Tensor MultiHeadAttention::forward(const Tensor& x, bool causal, float dropout_p) const {
    PROFILE("attn");
    // x: [T, C] where C = n_heads * head_dim; true per-head attention
    assert(n_embd_ % n_heads_ == 0);
    size_t T = x.shape[0];
    Tensor Q = x.matmul(Wq_);  // [T, C]
    Tensor K = x.matmul(Wk_);
    Tensor V = x.matmul(Wv_);
    if (bias_) {
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < n_embd_; ++j) {
                Q(i, j) += bq_.data[j];
                K(i, j) += bk_.data[j];
                V(i, j) += bv_.data[j];
            }
    }

    auto slice_head = [&](const Tensor& t, size_t h) -> Tensor {
        Tensor out({T, head_dim_}, 0.0f);
        size_t base = h * head_dim_;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < head_dim_; ++j) out(i, j) = t(i, base + j);
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
            // I84 FlashAttention-2 full prefill: online softmax, O(D+block) aux
            out_h = flash_attention_full(Qh, Kh, Vh, scale, causal, FLASH_BLOCK_SIZE);
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
                    for (size_t j = i + 1; j < T; ++j) scores(i, j) = -1e9f;
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
            for (size_t j = 0; j < head_dim_; ++j) out(i, base + j) = out_h(i, j);
    }

    Tensor proj = out.matmul(Wo_);  // [T, C]
    if (bias_) {
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < n_embd_; ++j) proj(i, j) += bo_.data[j];
    }
    return proj;
}

Tensor MultiHeadAttention::forward_incremental(const Tensor& x, KVCache& cache, size_t layer,
                                               size_t pos) const {
    assert(x.shape[0] == 1 && x.shape[1] == n_embd_);
    assert(n_embd_ % n_heads_ == 0);
    // Compute Q,K,V for single token
    Tensor Q = x.matmul(Wq_);  // [1, C]
    Tensor K = x.matmul(Wk_);
    Tensor V = x.matmul(Wv_);
    if (bias_) {
        for (size_t j = 0; j < n_embd_; ++j) {
            Q(0, j) += bq_.data[j];
            K(0, j) += bk_.data[j];
            V(0, j) += bv_.data[j];
        }
    }
    // Retrieve old cached K/V (size pos)
    Tensor old_K = cache.get_k_slice(layer);
    Tensor old_V = cache.get_v_slice(layer);
    // Update cache with new K/V at pos (cur_len should be pos)
    // Ensure cache size is pos before update; if not, set it
    if (cache.size() != pos) {
        cache.set_size(pos);
    }
    cache.update(layer, K, V);
    // Build K_all and V_all: concat old + new
    size_t K_len = pos + 1;
    Tensor K_all({K_len, n_embd_}, 0.0f);
    Tensor V_all({K_len, n_embd_}, 0.0f);
    for (size_t i = 0; i < pos && i < old_K.shape[0]; ++i) {
        for (size_t j = 0; j < n_embd_; ++j) {
            K_all(i, j) = old_K(i, j);
            V_all(i, j) = old_V(i, j);
        }
    }
    for (size_t j = 0; j < n_embd_; ++j) {
        K_all(pos, j) = K(0, j);
        V_all(pos, j) = V(0, j);
    }

    auto slice_head_all = [&](const Tensor& t, size_t h, size_t T) -> Tensor {
        Tensor out({T, head_dim_}, 0.0f);
        size_t base = h * head_dim_;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < head_dim_; ++j) out(i, j) = t(i, base + j);
        return out;
    };
    auto slice_head_q = [&](const Tensor& t, size_t h) -> Tensor {
        Tensor out({1, head_dim_}, 0.0f);
        size_t base = h * head_dim_;
        for (size_t j = 0; j < head_dim_; ++j) out(0, j) = t(0, base + j);
        return out;
    };

    Tensor out({1, n_embd_}, 0.0f);
    float scale = 1.0f / std::sqrt((float)head_dim_);
    for (size_t h = 0; h < n_heads_; ++h) {
        Tensor Qh = slice_head_q(Q, h);               // [1, Hd]
        Tensor Kh = slice_head_all(K_all, h, K_len);  // [K_len, Hd]
        Tensor Vh = slice_head_all(V_all, h, K_len);
        Tensor out_h;
        if (K_len > 128) {
            out_h = flash_attention_incremental(Qh, Kh, Vh, scale, FLASH_BLOCK_SIZE);
        } else {
            Tensor Kt = Kh.transpose();  // [Hd, K_len]
            Tensor scores = Qh.matmul(Kt);
            for (auto& v : scores.data) v *= scale;
            Tensor attn = scores.softmax(1);  // [1, K_len]
            out_h = attn.matmul(Vh);          // [1, Hd]
        }
        size_t base = h * head_dim_;
        for (size_t j = 0; j < head_dim_; ++j) out(0, base + j) = out_h(0, j);
    }
    Tensor proj = out.matmul(Wo_);  // [1, C]
    if (bias_) {
        for (size_t j = 0; j < n_embd_; ++j) proj(0, j) += bo_.data[j];
    }
    return proj;
}

Tensor MultiHeadAttention::backward(const Tensor& x, const Tensor& grad_out) const {
    // Manual backward mirroring forward — recompute intermediates
    size_t T = x.shape[0];
    Tensor Q = x.matmul(Wq_);
    Tensor K = x.matmul(Wk_);
    Tensor V = x.matmul(Wv_);
    if (bias_) {
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < n_embd_; ++j) {
                Q(i, j) += bq_.data[j];
                K(i, j) += bk_.data[j];
                V(i, j) += bv_.data[j];
            }
    }
    // grad for Wo / bo
    // Need out = concat(out_h) before Wo
    // Recompute out (pre-proj) for grad
    auto slice_head = [&](const Tensor& t, size_t h) -> Tensor {
        Tensor out({T, head_dim_}, 0.0f);
        size_t base = h * head_dim_;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < head_dim_; ++j) out(i, j) = t(i, base + j);
        return out;
    };
    auto unslice_to = [&](Tensor& base, const Tensor& slice, size_t h) {
        size_t b = h * head_dim_;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < head_dim_; ++j) base(i, b + j) = slice(i, j);
    };
    Tensor out({T, n_embd_}, 0.0f);
    // Store per-head intermediates for backward
    struct HeadCache {
        Tensor Qh, Kh, Vh, scores, attn, out_h;
    };
    std::vector<HeadCache> caches;
    caches.reserve(n_heads_);
    float scale = 1.0f / std::sqrt((float)head_dim_);
    for (size_t h = 0; h < n_heads_; ++h) {
        Tensor Qh = slice_head(Q, h), Kh = slice_head(K, h), Vh = slice_head(V, h);
        Tensor Kt = Kh.transpose();
        Tensor scores = Qh.matmul(Kt);
        for (auto& v : scores.data) v *= scale;
        for (size_t i = 0; i < T; ++i)
            for (size_t j = i + 1; j < T; ++j) scores(i, j) = -1e9f;
        Tensor attn = scores.softmax(1);
        Tensor out_h = attn.matmul(Vh);
        HeadCache c{Qh, Kh, Vh, scores, attn, out_h};
        caches.push_back(std::move(c));
        unslice_to(out, out_h, h);
    }
    // grad for Wo: dWo = out^T * grad_out  [C,C]
    Tensor dWo({n_embd_, n_embd_}, 0.0f);
    for (size_t i = 0; i < n_embd_; ++i)
        for (size_t j = 0; j < n_embd_; ++j) {
            float acc = 0;
            for (size_t t = 0; t < T; ++t) acc += out(t, i) * grad_out(t, j);
            dWo(i, j) = acc;
        }
    const_cast<Tensor&>(Wo_).add_grad(dWo);
    if (bias_) {
        Tensor dbo({n_embd_}, 0.0f);
        for (size_t j = 0; j < n_embd_; ++j) {
            float s = 0;
            for (size_t i = 0; i < T; ++i) s += grad_out(i, j);
            dbo.data[j] = s;
        }
        const_cast<Tensor&>(bo_).add_grad(dbo);
    }
    // dOut = grad_out * Wo^T  [T,C]
    Tensor dOut({T, n_embd_}, 0.0f);
    for (size_t i = 0; i < T; ++i)
        for (size_t j = 0; j < n_embd_; ++j) {
            float acc = 0;
            for (size_t k = 0; k < n_embd_; ++k)
                acc += grad_out(i, k) * Wo_.data[k * n_embd_ + j];  // Wo is [C,C], data row-major
            // Actually Wo(i,j) access via operator, but we can use direct
            // Use Wo(j,k) transpose: Wo^T(j,k)=Wo(k,j)
            // Simpler: recompute via loops with accessor
            acc = 0;
            for (size_t k = 0; k < n_embd_; ++k) acc += grad_out(i, k) * Wo_(j, k);
            dOut(i, j) = acc;
        }
    // Now per-head backward
    Tensor dQ({T, n_embd_}, 0.0f), dK({T, n_embd_}, 0.0f), dV({T, n_embd_}, 0.0f);
    for (size_t h = 0; h < n_heads_; ++h) {
        auto& c = caches[h];
        Tensor dOut_h = slice_head(dOut, h);
        // out_h = attn * Vh  => dAttn = dOut_h * Vh^T, dVh = attn^T * dOut_h
        Tensor dAttn({T, T}, 0.0f);
        for (size_t i = 0; i < T; ++i)
            for (size_t j = 0; j < T; ++j) {
                float acc = 0;
                for (size_t d = 0; d < head_dim_; ++d) acc += dOut_h(i, d) * c.Vh(j, d);
                dAttn(i, j) = acc;
            }
        Tensor dVh({T, head_dim_}, 0.0f);
        for (size_t i = 0; i < T; ++i)
            for (size_t d = 0; d < head_dim_; ++d) {
                float acc = 0;
                for (size_t j = 0; j < T; ++j) acc += c.attn(j, i) * dOut_h(j, d);  // attn^T
                // Actually dVh(j,d) = sum_i attn(i,j)*dOut_h(i,d) ??? Let's use correct: dVh =
                // attn^T * dOut_h So dVh(j,d) = sum_i attn(i,j)*dOut_h(i,d)
            }
        // Correct dVh
        for (size_t j = 0; j < T; ++j)
            for (size_t d = 0; d < head_dim_; ++d) {
                float acc = 0;
                for (size_t i = 0; i < T; ++i) acc += c.attn(i, j) * dOut_h(i, d);
                dVh(j, d) = acc;
            }
        // Softmax backward: dScores = attn * (dAttn - sum(dAttn*attn))
        Tensor dScores({T, T}, 0.0f);
        for (size_t i = 0; i < T; ++i) {
            float sum = 0;
            for (size_t j = 0; j < T; ++j) sum += dAttn(i, j) * c.attn(i, j);
            for (size_t j = 0; j < T; ++j) dScores(i, j) = c.attn(i, j) * (dAttn(i, j) - sum);
            for (size_t j = 0; j < T; ++j) dScores(i, j) *= scale;
            // causal mask: zero where j>i
            for (size_t j = i + 1; j < T; ++j) dScores(i, j) = 0;
        }
        Tensor dQh({T, head_dim_}, 0.0f), dKh({T, head_dim_}, 0.0f);
        for (size_t i = 0; i < T; ++i)
            for (size_t d = 0; d < head_dim_; ++d) {
                float accQ = 0, accK = 0;
                for (size_t j = 0; j < T; ++j) {
                    accQ += dScores(i, j) * c.Kh(j, d);
                    accK += dScores(j, i) * c.Qh(j, d);
                }
                dQh(i, d) = accQ;
                dKh(i, d) = accK;
            }
        unslice_to(dQ, dQh, h);
        unslice_to(dK, dKh, h);
        // dV already computed as dVh unsliced
        for (size_t i = 0; i < T; ++i)
            for (size_t d = 0; d < head_dim_; ++d) dV(i, h * head_dim_ + d) = dVh(i, d);
    }
    // Grad for Wq/Wk/Wv and bias, and dX
    // dWq = x^T * dQ  [C,C]
    auto matmul_grad_W = [&](const Tensor& X, const Tensor& dY, Tensor& W) {
        Tensor dW({W.shape[0], W.shape[1]}, 0.0f);
        for (size_t i = 0; i < W.shape[0]; ++i)
            for (size_t j = 0; j < W.shape[1]; ++j) {
                float acc = 0;
                for (size_t t = 0; t < T; ++t) acc += X(t, i) * dY(t, j);
                dW(i, j) = acc;
            }
        const_cast<Tensor&>(W).add_grad(dW);
    };
    matmul_grad_W(x, dQ, const_cast<Tensor&>(Wq_));
    matmul_grad_W(x, dK, const_cast<Tensor&>(Wk_));
    matmul_grad_W(x, dV, const_cast<Tensor&>(Wv_));
    if (bias_) {
        Tensor dbq({n_embd_}, 0.0f), dbk({n_embd_}, 0.0f), dbv({n_embd_}, 0.0f);
        for (size_t j = 0; j < n_embd_; ++j) {
            float s = 0;
            for (size_t i = 0; i < T; ++i) s += dQ(i, j);
            dbq.data[j] = s;
            s = 0;
            for (size_t i = 0; i < T; ++i) s += dK(i, j);
            dbk.data[j] = s;
            s = 0;
            for (size_t i = 0; i < T; ++i) s += dV(i, j);
            dbv.data[j] = s;
        }
        const_cast<Tensor&>(bq_).add_grad(dbq);
        const_cast<Tensor&>(bk_).add_grad(dbk);
        const_cast<Tensor&>(bv_).add_grad(dbv);
    }
    // dX = dQ*Wq^T + dK*Wk^T + dV*Wv^T
    Tensor dX({T, n_embd_}, 0.0f);
    for (size_t i = 0; i < T; ++i)
        for (size_t j = 0; j < n_embd_; ++j) {
            float acc = 0;
            for (size_t k = 0; k < n_embd_; ++k) acc += dQ(i, k) * Wq_(j, k);
            for (size_t k = 0; k < n_embd_; ++k) acc += dK(i, k) * Wk_(j, k);
            for (size_t k = 0; k < n_embd_; ++k) acc += dV(i, k) * Wv_(j, k);
            dX(i, j) = acc;
        }
    return dX;
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
    if (bias_) return {&Wq_, &Wk_, &Wv_, &Wo_, &bq_, &bk_, &bv_, &bo_};
    return {&Wq_, &Wk_, &Wv_, &Wo_};
}
std::vector<const Tensor*> MultiHeadAttention::parameters() const {
    if (bias_) return {&Wq_, &Wk_, &Wv_, &Wo_, &bq_, &bk_, &bv_, &bo_};
    return {&Wq_, &Wk_, &Wv_, &Wo_};
}

}  // namespace llm
