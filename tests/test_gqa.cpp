#include "llm/mqa.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
    const size_t T = 4, C = 16, nq = 4, hd = 4;
    llm::Tensor x({T, C}, 0.0f);
    for (size_t i = 0; i < x.data.size(); ++i) x.data[i] = (float)(i % 11) * 0.1f - 0.4f;
    llm::Tensor Wq({C, nq * hd}, 0.0f), Wk({C, nq * hd}, 0.0f), Wv({C, nq * hd}, 0.0f);
    for (auto& v : Wq.data) v = 0.1f; for (auto& v : Wk.data) v = -0.07f; for (auto& v : Wv.data) v = 0.05f;
    // n_kv == n_q must match full per-head path (self-consistency vs second call)
    auto y1 = llm::gqa_forward(x, Wq, Wk, Wv, nq, nq, hd);
    auto y2 = llm::gqa_forward(x, Wq, Wk, Wv, nq, nq, hd);
    assert(y1.data == y2.data);
    for (auto v : y1.data) assert(std::isfinite(v));
    // n_kv == 1 must match mqa_forward with stacked Wkv=[Wk_single, Wv_single]
    llm::Tensor Wk1({C, hd}, 0.0f), Wv1({C, hd}, 0.0f);
    for (size_t i = 0; i < Wk1.data.size(); ++i) { Wk1.data[i] = Wk.data[i]; Wv1.data[i] = Wv.data[i]; }
    llm::Tensor Wkv({C, 2 * hd}, 0.0f);
    for (size_t r = 0; r < C; ++r)
        for (size_t d = 0; d < hd; ++d) {
            Wkv(r, d) = Wk1(r, d);
            Wkv(r, hd + d) = Wv1(r, d);
        }
    auto ymqa = llm::mqa_forward(x, Wq, Wkv, nq, hd);
    // gqa with replicated kv weights: build Wk_rep/Wv_rep by tiling single head nq times? No —
    // gqa nkv=1 uses the single head directly; construct equivalent full-size Wk/Wv by repeating.
    // Instead compare gqa(nkv=1, Wk1, Wv1) vs mqa with same single-head weights:
    auto ygqa1 = llm::gqa_forward(x, Wq, Wk1, Wv1, nq, 1, hd);
    float md = 0;
    for (size_t i = 0; i < ymqa.data.size(); ++i) md = std::max(md, std::fabs(ymqa.data[i] - ygqa1.data[i]));
    std::cout << "mqa-vs-gqa1 maxd=" << md << "\n";
    assert(md < 1e-4);
    std::cout << "gqa test passed\n";
    return 0;
}
