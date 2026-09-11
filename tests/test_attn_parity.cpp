#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/attention.h"
#include "llm/flash_attention.h"
int main() {
    // Full vs incremental equivalence (T=8, small -> naive path)
    llm::MultiHeadAttention attn(16, 4, 32, false);
    llm::Tensor x({8, 16}, 0.0f);
    for (size_t i = 0; i < x.data.size(); ++i) x.data[i] = (float)(i % 13) * 0.1f - 0.5f;
    llm::Tensor full = attn.forward(x, true, 0.0f);
    llm::KVCache cache(2, 16, 32);
    llm::Tensor last;
    for (size_t pos = 0; pos < 8; ++pos) {
        llm::Tensor tok({1, 16}, 0.0f);
        for (size_t j = 0; j < 16; ++j) tok(0, j) = x.data[pos * 16 + j];
        last = attn.forward_incremental(tok, cache, 0, pos);
    }
    float maxd = 0;
    for (size_t j = 0; j < 16; ++j)
        maxd = std::max(maxd, std::fabs(full.data[7 * 16 + j] - last.data[j]));
    std::cout << "full-vs-incr maxd=" << maxd << "\n";
    assert(maxd < 1e-3);
    // Flash incremental vs naive for K_len=150 (>128 dispatch boundary)
    llm::Tensor Q({1, 8}, 0.0f), K({150, 8}, 0.0f), V({150, 8}, 0.0f);
    for (auto& v : Q.data) v = 0.2f;
    for (size_t i = 0; i < K.data.size(); ++i) K.data[i] = (float)(i % 7) * 0.1f;
    for (size_t i = 0; i < V.data.size(); ++i) V.data[i] = (float)(i % 5) * 0.2f - 0.3f;
    float scale = 1.0f / std::sqrt(8.0f);
    auto y_flash = llm::flash_attention_incremental(Q, K, V, scale, 32);
    // naive reference
    auto Kt = K.transpose();
    auto scores = Q.matmul(Kt);
    for (auto& v : scores.data) v *= scale;
    auto a = scores.softmax(1);
    auto y_ref = a.matmul(V);
    float md2 = 0;
    for (size_t i = 0; i < y_flash.data.size(); ++i)
        md2 = std::max(md2, std::fabs(y_flash.data[i] - y_ref.data[i]));
    std::cout << "flash-vs-naive maxd=" << md2 << "\n";
    assert(md2 < 1e-4);
    // T=33 and T=129 boundary spot-checks (causal full path incl. flash T>128)
    for (size_t T : {33, 129}) {
        llm::Tensor xb({T, 16}, 0.1f);
        auto yb = attn.forward(xb, true, 0.0f);
        assert(yb.shape[0] == T);
        for (auto v : yb.data) assert(std::isfinite(v));
    }
    std::cout << "attn_parity test passed\n";
    return 0;
}
