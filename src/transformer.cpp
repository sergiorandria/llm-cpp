#include "llm/transformer.h"

#include "llm/fused.h"

namespace llm {

FeedForward::FeedForward(size_t n_embd, size_t hidden_dim, bool bias, Activation act)
    : W1_({n_embd, hidden_dim ? hidden_dim : 4 * n_embd}),
      W2_({hidden_dim ? hidden_dim : 4 * n_embd, n_embd}),
      W3_({n_embd, hidden_dim ? hidden_dim : 4 * n_embd}),
      b1_({hidden_dim ? hidden_dim : 4 * n_embd}),
      b2_({n_embd}) {
    (void)bias;
    act_ = act;
    W1_.randn(0, 0.02f);
    W2_.randn(0, 0.02f);
    if (act_ == Activation::SILU) W3_.randn(0, 0.02f);
}

Tensor FeedForward::forward(const Tensor& x) const {
    if (act_ == Activation::SILU) {
        Tensor h1 = x.matmul(W1_);
        Tensor h3 = x.matmul(W3_);
        for (size_t i = 0; i < h1.data.size(); ++i)
            h1.data[i] = h1.data[i] / (1.0f + std::exp(-h1.data[i]));
        Tensor gated = h1.mul(h3);
        Tensor out = gated.matmul(W2_);
        return out;
    }
    Tensor h = x.matmul(W1_);
    for (auto& v : h.data) {
        v = 0.5f * v *
            (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265f) * (v + 0.044715f * v * v * v)));
    }
    Tensor out = h.matmul(W2_);
    return out;
}
Tensor FeedForward::backward(const Tensor& x, const Tensor& grad_out) const {
    size_t T = x.shape[0];
    if (act_ == Activation::SILU) {
        Tensor h1 = x.matmul(W1_);
        Tensor h3 = x.matmul(W3_);
        Tensor silu_h1(h1.shape, 0.0f);
        for (size_t i = 0; i < h1.data.size(); ++i) {
            float v = h1.data[i];
            float s = 1.0f / (1.0f + std::exp(-v));
            silu_h1.data[i] = v * s;
        }
        Tensor gated = silu_h1.mul(h3);
        // grad for W2: gated^T * grad_out
        Tensor dW2({gated.shape[1], grad_out.shape[1]}, 0.0f);
        for (size_t i = 0; i < gated.shape[1]; ++i)
            for (size_t j = 0; j < grad_out.shape[1]; ++j) {
                float acc = 0;
                for (size_t t = 0; t < T; ++t) acc += gated(t, i) * grad_out(t, j);
                dW2(i, j) = acc;
            }
        const_cast<Tensor&>(W2_).add_grad(dW2);
        // grad_b2
        Tensor db2({grad_out.shape[1]}, 0.0f);
        for (size_t j = 0; j < grad_out.shape[1]; ++j) {
            float s = 0;
            for (size_t t = 0; t < T; ++t) s += grad_out(t, j);
            db2.data[j] = s;
        }
        const_cast<Tensor&>(b2_).add_grad(db2);
        // dGated = grad_out * W2^T
        Tensor dGated({T, gated.shape[1]}, 0.0f);
        for (size_t t = 0; t < T; ++t)
            for (size_t i = 0; i < gated.shape[1]; ++i) {
                float acc = 0;
                for (size_t j = 0; j < grad_out.shape[1]; ++j) acc += grad_out(t, j) * W2_(i, j);
                dGated(t, i) = acc;
            }
        // dGated splits to silu_h1 and h3 via mul
        Tensor dSilu_h1(h1.shape, 0.0f), dH3(h3.shape, 0.0f);
        for (size_t i = 0; i < dGated.data.size(); ++i) {
            dSilu_h1.data[i] = dGated.data[i] * h3.data[i];
            dH3.data[i] = dGated.data[i] * silu_h1.data[i];
        }
        // silu backward: dH1 = dSilu * silu'(h1), silu' = sigmoid*(1 + h1*(1-sigmoid))
        Tensor dH1(h1.shape, 0.0f);
        for (size_t i = 0; i < h1.data.size(); ++i) {
            float v = h1.data[i];
            float s = 1.0f / (1.0f + std::exp(-v));
            float ds = s * (1 + v * (1 - s));
            dH1.data[i] = dSilu_h1.data[i] * ds;
        }
        // dW1 = x^T * dH1, dW3 = x^T * dH3
        auto dW_from = [&](const Tensor& dH, Tensor& W) {
            Tensor dW({W.shape[0], W.shape[1]}, 0.0f);
            for (size_t i = 0; i < W.shape[0]; ++i)
                for (size_t j = 0; j < W.shape[1]; ++j) {
                    float acc = 0;
                    for (size_t t = 0; t < T; ++t) acc += x(t, i) * dH(t, j);
                    dW(i, j) = acc;
                }
            const_cast<Tensor&>(W).add_grad(dW);
        };
        dW_from(dH1, const_cast<Tensor&>(W1_));
        dW_from(dH3, const_cast<Tensor&>(W3_));
        Tensor db1({dH1.shape[1]}, 0.0f);
        for (size_t j = 0; j < dH1.shape[1]; ++j) {
            float s = 0;
            for (size_t t = 0; t < T; ++t) s += dH1(t, j);
            db1.data[j] = s;
        }
        const_cast<Tensor&>(b1_).add_grad(db1);
        // dX = dH1*W1^T + dH3*W3^T
        Tensor dX({T, x.shape[1]}, 0.0f);
        for (size_t t = 0; t < T; ++t)
            for (size_t i = 0; i < x.shape[1]; ++i) {
                float acc = 0;
                for (size_t j = 0; j < dH1.shape[1]; ++j) acc += dH1(t, j) * W1_(i, j);
                for (size_t j = 0; j < dH3.shape[1]; ++j) acc += dH3(t, j) * W3_(i, j);
                dX(t, i) = acc;
            }
        return dX;
    } else {
        Tensor h = x.matmul(W1_);
        Tensor h_gelu(h.shape, 0.0f);
        for (size_t i = 0; i < h.data.size(); ++i) {
            float v = h.data[i];
            h_gelu.data[i] =
                0.5f * v *
                (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265f) * (v + 0.044715f * v * v * v)));
        }
        Tensor dW2({h.shape[1], grad_out.shape[1]}, 0.0f);
        for (size_t i = 0; i < h.shape[1]; ++i)
            for (size_t j = 0; j < grad_out.shape[1]; ++j) {
                float acc = 0;
                for (size_t t = 0; t < T; ++t) acc += h_gelu(t, i) * grad_out(t, j);
                dW2(i, j) = acc;
            }
        const_cast<Tensor&>(W2_).add_grad(dW2);
        Tensor db2({grad_out.shape[1]}, 0.0f);
        for (size_t j = 0; j < grad_out.shape[1]; ++j) {
            float s = 0;
            for (size_t t = 0; t < T; ++t) s += grad_out(t, j);
            db2.data[j] = s;
        }
        const_cast<Tensor&>(b2_).add_grad(db2);
        Tensor dGelu({T, h.shape[1]}, 0.0f);
        for (size_t t = 0; t < T; ++t)
            for (size_t i = 0; i < h.shape[1]; ++i) {
                float acc = 0;
                for (size_t j = 0; j < grad_out.shape[1]; ++j) acc += grad_out(t, j) * W2_(i, j);
                dGelu(t, i) = acc;
            }
        Tensor dH(h.shape, 0.0f);
        for (size_t i = 0; i < h.data.size(); ++i) {
            float v = h.data[i];
            float th = std::tanh(std::sqrt(2.0f / 3.14159265f) * (v + 0.044715f * v * v * v));
            float sech2 = 1 - th * th;
            float inner = std::sqrt(2.0f / 3.14159265f) * (1 + 3 * 0.044715f * v * v);
            float dg = 0.5f * (1 + th) + 0.5f * v * sech2 * inner;
            dH.data[i] = dGelu.data[i] * dg;
        }
        Tensor dW1({W1_.shape[0], W1_.shape[1]}, 0.0f);
        for (size_t i = 0; i < W1_.shape[0]; ++i)
            for (size_t j = 0; j < W1_.shape[1]; ++j) {
                float acc = 0;
                for (size_t t = 0; t < T; ++t) acc += x(t, i) * dH(t, j);
                dW1(i, j) = acc;
            }
        const_cast<Tensor&>(W1_).add_grad(dW1);
        Tensor db1({dH.shape[1]}, 0.0f);
        for (size_t j = 0; j < dH.shape[1]; ++j) {
            float s = 0;
            for (size_t t = 0; t < T; ++t) s += dH(t, j);
            db1.data[j] = s;
        }
        const_cast<Tensor&>(b1_).add_grad(db1);
        Tensor dX({T, x.shape[1]}, 0.0f);
        for (size_t t = 0; t < T; ++t)
            for (size_t i = 0; i < x.shape[1]; ++i) {
                float acc = 0;
                for (size_t j = 0; j < dH.shape[1]; ++j) acc += dH(t, j) * W1_(i, j);
                dX(t, i) = acc;
            }
        return dX;
    }
}

TransformerBlock::TransformerBlock(size_t n_embd, size_t n_heads, size_t block_size,
                                   TransformerConfig cfg)
    : attn_(n_embd, n_heads, block_size),
      ffn_(n_embd),  // cfg unused yet
      // cfg dropout stored if needed

      ln1_gamma_({n_embd}, 1.0f),
      ln1_beta_({n_embd}, 0.0f),
      ln2_gamma_({n_embd}, 1.0f),
      ln2_beta_({n_embd}, 0.0f) {}

Tensor TransformerBlock::forward(const Tensor& x) const {
    Tensor ln1 = x.layernorm(&ln1_gamma_, &ln1_beta_);
    Tensor attn_out = attn_.forward(ln1);
    Tensor y(x.shape, 0.0f);
    for (size_t i = 0; i < y.data.size(); ++i) y.data[i] = x.data[i] + attn_out.data[i];
    Tensor ln2 = y.layernorm(&ln2_gamma_, &ln2_beta_);
    Tensor ffn_out = ffn_.forward(ln2);
    Tensor out(y.shape, 0.0f);
    for (size_t i = 0; i < out.data.size(); ++i) out.data[i] = y.data[i] + ffn_out.data[i];
    return out;
}
Tensor TransformerBlock::backward(const Tensor& x, const Tensor& grad_out) const {
    // Recompute forward intermediates for backward
    Tensor ln1 = x.layernorm(&ln1_gamma_, &ln1_beta_);
    Tensor attn_out = attn_.forward(ln1);
    Tensor y(x.shape, 0.0f);
    for (size_t i = 0; i < y.data.size(); ++i) y.data[i] = x.data[i] + attn_out.data[i];
    Tensor ln2 = y.layernorm(&ln2_gamma_, &ln2_beta_);
    Tensor ffn_out = ffn_.forward(ln2);
    // out = y + ffn_out, so dY and dFFN both = grad_out, dY also gets residual
    Tensor dY = grad_out;  // will accumulate ffn backward's dY
    Tensor dFFN = ffn_.backward(ln2, grad_out);
    // dY already = grad_out, plus grad from ffn's input (ln2) will be added via layernorm backward
    // later For residual y = x + attn, and out = y + ffn, dY total = grad_out (from residual) +
    // dFFN's grad to y via ln2 Actually ffn backward returns dLn2, then layernorm backward gives
    // dY_2 Simplify: dLn2 = dFFN (output of ffn backward is dLn2)
    Tensor dLn2 = dFFN;
    auto ln2_bwd = y.layernorm_backward(dLn2, &ln2_gamma_);
    Tensor dY_from_ln2 = ln2_bwd.grad_x;
    const_cast<Tensor&>(ln2_gamma_).add_grad(ln2_bwd.grad_gamma);
    const_cast<Tensor&>(ln2_beta_).add_grad(ln2_bwd.grad_beta);
    Tensor dY_total = dY.add(dY_from_ln2);
    // y = x + attn_out, so dX and dAttn both = dY_total, dLn1 comes from attn
    Tensor dAttn = dY_total;
    Tensor dX_attn = attn_.backward(ln1, dAttn);
    auto ln1_bwd = x.layernorm_backward(dX_attn, &ln1_gamma_);
    Tensor dX_from_ln1 = ln1_bwd.grad_x;
    const_cast<Tensor&>(ln1_gamma_).add_grad(ln1_bwd.grad_gamma);
    const_cast<Tensor&>(ln1_beta_).add_grad(ln1_bwd.grad_beta);
    Tensor dX = dY_total.add(dX_from_ln1);
    return dX;
}

std::vector<Tensor*> FeedForward::parameters() {
    if (act_ == Activation::SILU) return {&W1_, &W2_, &W3_, &b1_, &b2_};
    return {&W1_, &W2_, &b1_, &b2_};
}
std::vector<const Tensor*> FeedForward::parameters() const {
    if (act_ == Activation::SILU) return {&W1_, &W2_, &W3_, &b1_, &b2_};
    return {&W1_, &W2_, &b1_, &b2_};
}
std::vector<Tensor*> TransformerBlock::parameters() {
    std::vector<Tensor*> p;
    auto ap = attn_.parameters();
    p.insert(p.end(), ap.begin(), ap.end());
    auto fp = ffn_.parameters();
    p.insert(p.end(), fp.begin(), fp.end());
    p.push_back(&ln1_gamma_);
    p.push_back(&ln1_beta_);
    p.push_back(&ln2_gamma_);
    p.push_back(&ln2_beta_);
    return p;
}
std::vector<const Tensor*> TransformerBlock::parameters() const {
    std::vector<const Tensor*> p;
    auto ap = attn_.parameters();
    p.insert(p.end(), ap.begin(), ap.end());
    auto fp = ffn_.parameters();
    p.insert(p.end(), fp.begin(), fp.end());
    p.push_back(&ln1_gamma_);
    p.push_back(&ln1_beta_);
    p.push_back(&ln2_gamma_);
    p.push_back(&ln2_beta_);
    return p;
}

}  // namespace llm
