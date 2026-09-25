// tests/test_mamba_backward.cpp — M10: gradient check + overfit for Mamba path.
// 1. Finite-difference check of MambaBlock::backward (params + input).
// 2. GPT(use_mamba) backward yields finite non-zero grads; a few SGD steps
//    on a tiny batch reduce the loss (overfit smoke).
#include <cmath>
#include <iostream>
#include <vector>

#include "llm/loss.h"
#include "llm/mamba.h"
#include "llm/model.h"

static float loss_of(llm::MambaBlock& blk, const llm::Tensor& x) {
    auto out = blk.forward(x);
    float s = 0;
    for (float v : out.data) s += v;
    return s;
}

int main() {
    using namespace llm;
    const size_t E = 8, D = 8, N = 4, R = 2, K = 2, T = 4;
    MambaBlock blk(E, D, N, R, K, true);
    Tensor x({T, E}, 0.0f);
    for (size_t i = 0; i < x.data.size(); ++i) x.data[i] = 0.1f * (float)((i % 7) + 1);

    // Analytic grads with dL/dout = ones (loss = sum(out)).
    Tensor gout({T, E}, 1.0f);
    // zero param grads
    for (auto* p : blk.parameters()) p->zero_grad();
    Tensor dx = blk.backward(x, gout);

    const float eps = 1e-3f;
    float max_err = 0;
    auto params = blk.parameters();
    for (auto* p : params) {
        for (size_t i = 0; i < p->data.size(); ++i) {
            float old = p->data[i];
            p->data[i] = old + eps;
            float lp = loss_of(blk, x);
            p->data[i] = old - eps;
            float lm = loss_of(blk, x);
            p->data[i] = old;
            float num = (lp - lm) / (2 * eps);
            float ana = p->grad[i];
            float err = std::abs(num - ana);
            // scale-aware tolerance (mirrors test_rmsnorm style)
            float tol = 2e-2f + 2e-2f * std::abs(num);
            if (err > tol) {
                std::cerr << "FAIL: param grad mismatch idx=" << i << " num=" << num
                          << " ana=" << ana << " err=" << err << "\n";
                return 1;
            }
            max_err = std::max(max_err, err);
        }
    }
    // Input grad check (subset of entries for speed).
    for (size_t i = 0; i < x.data.size(); i += 3) {
        float old = x.data[i];
        x.data[i] = old + eps;
        float lp = loss_of(blk, x);
        x.data[i] = old - eps;
        float lm = loss_of(blk, x);
        x.data[i] = old;
        float num = (lp - lm) / (2 * eps);
        float err = std::abs(num - dx.data[i]);
        float tol = 2e-2f + 2e-2f * std::abs(num);
        if (err > tol) {
            std::cerr << "FAIL: input grad mismatch idx=" << i << " num=" << num
                      << " ana=" << dx.data[i] << "\n";
            return 1;
        }
        max_err = std::max(max_err, err);
    }
    std::cout << "mamba block grad check max_err=" << max_err << "\n";

    // GPT mamba path: grads finite + non-zero, SGD overfit smoke.
    Config cfg;
    cfg.vocab_size = 16;
    cfg.n_embd = 8;
    cfg.n_heads = 2;
    cfg.n_layers = 1;
    cfg.block_size = 8;
    cfg.weight_tying = false;
    cfg.use_mamba = true;
    cfg.d_inner = 8;
    cfg.d_state = 4;
    cfg.dt_rank = 2;
    cfg.conv_kernel = 2;
    GPT m(cfg);
    std::vector<int> toks = {1, 2, 3, 4};
    m.zero_grad();
    auto [logits, hidden] = m.forward_with_hidden(toks);
    float l0 = compute_loss(logits, toks);
    Tensor dl = cross_entropy_backward(logits, toks);
    m.backward(dl, toks, hidden);
    {
        auto ps = m.parameters();
        double n2 = 0;
        for (auto* p : ps) {
            if (p->grad.size() != p->data.size()) {
                std::cerr << "FAIL: param grad missing\n";
                return 1;
            }
            for (float v : p->grad) {
                if (!std::isfinite(v)) {
                    std::cerr << "FAIL: non-finite mamba grad\n";
                    return 1;
                }
                n2 += (double)v * v;
            }
        }
        if (!(n2 > 0)) {
            std::cerr << "FAIL: all mamba grads zero\n";
            return 1;
        }
        std::cout << "mamba gpt grad norm=" << std::sqrt(n2) << " loss=" << l0 << "\n";
    }
    // 20 SGD steps, loss must decrease.
    const float lr = 0.05f;
    float prev = l0;
    for (int s = 0; s < 20; ++s) {
        m.zero_grad();
        auto [lg, hd] = m.forward_with_hidden(toks);
        float loss = compute_loss(lg, toks);
        Tensor d = cross_entropy_backward(lg, toks);
        m.backward(d, toks, hd);
        auto ps = m.parameters();
        for (auto* p : ps) {
            if (p->grad.size() != p->data.size()) continue;
            for (size_t i = 0; i < p->data.size(); ++i) p->data[i] -= lr * p->grad[i];
        }
        if (!std::isfinite(loss)) {
            std::cerr << "FAIL: non-finite loss at step " << s << "\n";
            return 1;
        }
        prev = loss;
    }
    auto lf = m.forward(toks);
    float l1 = compute_loss(lf, toks);
    std::cout << "mamba overfit: " << l0 << " -> " << l1 << "\n";
    if (!(l1 < prev) || !(l1 < l0)) {
        std::cerr << "FAIL: mamba loss did not decrease\n";
        return 1;
    }
    std::cout << "test_mamba_backward passed\n";
    return 0;
}
