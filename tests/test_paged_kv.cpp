#include "llm/kv_cache.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
    const size_t L = 2, T = 40, C = 8;
    llm::KVCacheConfig cc; cc.paged = false;
    llm::KVCacheConfig pc; pc.paged = true; pc.page_size = 16;
    llm::KVCache a(L, 64, C, cc), b(L, 64, C, pc);
    for (size_t t = 0; t < T; ++t) {
        llm::Tensor k({1, C}, 0.0f), v({1, C}, 0.0f);
        for (size_t j = 0; j < C; ++j) { k(0,j) = (float)t * 0.1f + j; v(0,j) = (float)t - (float)j * 0.5f; }
        for (size_t l = 0; l < L; ++l) { a.update(l, k, v); b.update(l, k, v); }
        a.advance(1); b.advance(1);
    }
    float md = 0;
    for (size_t l = 0; l < L; ++l) {
        auto ka = a.get_k_slice(l), kb = b.get_k_slice(l);
        auto va = a.get_v_slice(l), vb = b.get_v_slice(l);
        for (size_t i = 0; i < ka.data.size(); ++i) md = std::max(md, std::fabs(ka.data[i]-kb.data[i]));
        for (size_t i = 0; i < va.data.size(); ++i) md = std::max(md, std::fabs(va.data[i]-vb.data[i]));
    }
    std::cout << "paged-vs-contig maxd=" << md << " pages=" << b.num_pages(0) << "\n";
    assert(md == 0.0f);
    assert(b.num_pages(0) == 3);  // 40 tokens / 16 = 3 pages
    // evict + reuse: interleaved sequences don't accumulate pages
    b.evict();
    assert(b.num_pages(0) == 0 && b.size() == 40);  // storage freed, len kept until clear
    b.clear();
    assert(b.size() == 0);
    for (int s = 0; s < 10; ++s) {
        llm::Tensor k({1, C}, 1.0f), v({1, C}, 2.0f);
        b.update(0, k, v); b.advance(1);
        b.evict(); b.clear();
    }
    assert(b.num_pages(0) == 0);
    std::cout << "paged test passed\n";
    return 0;
}
