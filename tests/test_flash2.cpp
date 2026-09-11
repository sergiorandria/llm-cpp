#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/flash_attention.h"
int main() {
    const size_t T = 40, D = 8;
    llm::Tensor Q({T, D}, 0.0f), K({T, D}, 0.0f), V({T, D}, 0.0f);
    for (size_t i = 0; i < Q.data.size(); ++i) Q.data[i] = (float)(i % 7) * 0.15f - 0.4f;
    for (size_t i = 0; i < K.data.size(); ++i) K.data[i] = (float)(i % 5) * 0.2f - 0.3f;
    for (size_t i = 0; i < V.data.size(); ++i) V.data[i] = (float)(i % 9) * 0.1f - 0.2f;
    float scale = 1.0f / std::sqrt((float)D);
    for (bool causal : {true, false}) {
        auto y_new = llm::flash_attention_full(Q, K, V, scale, causal, 16);
        auto y_old = llm::flash_attention(Q, K, V, scale, causal, 16);
        // naive reference
        auto Kt = K.transpose();
        auto S = Q.matmul(Kt);
        for (auto& v : S.data) v *= scale;
        if (causal)
            for (size_t i = 0; i < T; ++i)
                for (size_t j = i + 1; j < T; ++j) S(i, j) = -1e9f;
        auto A = S.softmax(1);
        auto y_ref = A.matmul(V);
        float m1 = 0, m2 = 0;
        for (size_t i = 0; i < y_new.data.size(); ++i) {
            m1 = std::max(m1, std::fabs(y_new.data[i] - y_ref.data[i]));
            m2 = std::max(m2, std::fabs(y_new.data[i] - y_old.data[i]));
        }
        std::cout << "causal=" << causal << " fa2-vs-naive=" << m1 << " fa2-vs-fa1=" << m2 << "\n";
        assert(m1 < 1e-4f && m2 < 1e-4f);
    }
    std::cout << "flash2 test passed\n";
    return 0;
}
