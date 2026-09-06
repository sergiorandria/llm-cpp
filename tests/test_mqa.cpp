#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/mqa.h"

int main() {
    size_t T = 4, n_embd = 8, n_heads = 2, head_dim = 4;
    llm::Tensor x({T, n_embd}, 0.0f);
    x.randn(0, 0.02f);
    llm::Tensor Wq({n_embd, n_heads * head_dim});
    llm::Tensor Wkv({n_embd, 2 * head_dim});
    Wq.randn(0, 0.02f);
    Wkv.randn(0, 0.02f);
    auto out = llm::mqa_forward(x, Wq, Wkv, n_heads, head_dim);
    assert(out.shape[0] == T);
    assert(out.shape[1] == n_heads * head_dim);
    for (float v : out.data) assert(!std::isnan(v) && !std::isinf(v));
    // Check that different heads produce different outputs (since Q differs but K/V shared, still
    // different)
    float diff_heads = 0;
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < head_dim; ++d)
            diff_heads += std::abs(out(t, d) - out(t, head_dim + d));
    }
    std::cout << "mqa inter-head diff " << diff_heads << "\n";
    assert(diff_heads > 1e-6f);
    std::cout << "mqa test passed\n";
    return 0;
}
