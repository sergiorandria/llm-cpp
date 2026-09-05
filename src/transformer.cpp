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
    act_=act;
    W1_.randn(0, 0.02f); W2_.randn(0, 0.02f);
    if(act_==Activation::SILU) W3_.randn(0,0.02f);
}

Tensor FeedForward::forward(const Tensor& x) const {
    if(act_ == Activation::SILU){
        // SwiGLU: silu(x @ W1) * (x @ W3) @ W2  (gated)
        Tensor h1 = x.matmul(W1_);
        Tensor h3 = x.matmul(W3_);
        // silu on h1
        for(size_t i=0;i<h1.data.size();++i) h1.data[i] = h1.data[i] / (1.0f + std::exp(-h1.data[i]));
        // gate
        Tensor gated = h1.mul(h3);
        Tensor out = gated.matmul(W2_);
        return out;
    }
    Tensor h = x.matmul(W1_); // [T, 4*C]
    // GELU approx (tanh)
    for (auto& v : h.data) {
        v = 0.5f * v * (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265f) * (v + 0.044715f * v * v * v)));
    }
    Tensor out = h.matmul(W2_); // [T, C]
    return out;
}

TransformerBlock::TransformerBlock(size_t n_embd, size_t n_heads, size_t block_size, TransformerConfig cfg)
    : attn_(n_embd, n_heads, block_size),
      ffn_(n_embd), // cfg unused yet
      // cfg dropout stored if needed

      ln1_gamma_({n_embd}, 1.0f), ln1_beta_({n_embd}, 0.0f),
      ln2_gamma_({n_embd}, 1.0f), ln2_beta_({n_embd}, 0.0f) {}

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
    auto ap = attn_.parameters(); p.insert(p.end(), ap.begin(), ap.end());
    auto fp = ffn_.parameters(); p.insert(p.end(), fp.begin(), fp.end());
    p.push_back(&ln1_gamma_); p.push_back(&ln1_beta_);
    p.push_back(&ln2_gamma_); p.push_back(&ln2_beta_);
    return p;
}
std::vector<const Tensor*> TransformerBlock::parameters() const {
    std::vector<const Tensor*> p;
    auto ap = attn_.parameters(); p.insert(p.end(), ap.begin(), ap.end());
    auto fp = ffn_.parameters(); p.insert(p.end(), fp.begin(), fp.end());
    p.push_back(&ln1_gamma_); p.push_back(&ln1_beta_);
    p.push_back(&ln2_gamma_); p.push_back(&ln2_beta_);
    return p;
}

} // namespace llm
