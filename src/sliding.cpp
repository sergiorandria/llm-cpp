#include "llm/sliding.h"

#include <cmath>

namespace llm {

Tensor sliding_attention(const Tensor& Q, const Tensor& K, const Tensor& V, size_t window) {
    // Q,K,V: [T, C] where C = head_dim * maybe n_heads? For sliding we treat as single head
    // window = 512 default, sliding window attention: each query attends to last `window` keys
    assert(Q.shape.size() == 2 && K.shape.size() == 2 && V.shape.size() == 2);
    assert(Q.shape[0] == K.shape[0] && Q.shape[0] == V.shape[0]);
    assert(Q.shape[1] == K.shape[1] && Q.shape[1] == V.shape[1]);
    size_t T = Q.shape[0];
    size_t C = Q.shape[1];
    float scale = 1.0f / std::sqrt(float(C));
    Tensor Kt = K.transpose();     // [C, T]
    Tensor scores = Q.matmul(Kt);  // [T, T]
    for (auto& v : scores.data) v *= scale;
    // Apply causal + sliding window mask
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < T; ++j) {
            if (j > i) {
                scores(i, j) = -1e9f;
            } else if (i - j >= window) {
                scores(i, j) = -1e9f;
            }
        }
    }
    Tensor attn = scores.softmax(1);  // softmax per row
    Tensor out = attn.matmul(V);      // [T, C]
    return out;
}

}  // namespace llm
