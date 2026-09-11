// I85/I86/I90: pool reuse + 64B alignment + paged-KV pool integration.
#include "llm/kv_cache.h"
#include "llm/pool.h"
#include "llm/tensor.h"
#include <cassert>
#include <iostream>
int main() {
    // I90: every Tensor storage 64B-aligned
    llm::Tensor t({16, 16}, 1.0f);
    assert(!t.data.empty() && llm::is_aligned64(t.data.data()));
    llm::Tensor g({8}, 0.0f);
    g.zero_grad();
    assert(llm::is_aligned64(g.grad.data()));
    // numerics unchanged after allocator migration
    llm::Tensor A({2, 2}, 0.0f), B({2, 2}, 0.0f);
    A.data = {1, 2, 3, 4}; B.data = {5, 6, 7, 8};
    auto C = A.matmul(B);
    assert(C.data[0] == 19 && C.data[1] == 22 && C.data[2] == 43 && C.data[3] == 50);
    // I86: acquire/release reuse (no realloc on 2nd acquire of same numel)
    llm::TensorPool pool;
    llm::Tensor p1 = pool.acquire({8, 8});
    size_t m0 = pool.misses();
    assert(m0 == 1 && pool.hits() == 0);
    const float* ptr1 = p1.data.data();
    pool.release(std::move(p1));
    assert(pool.cached() == 1);
    llm::Tensor p2 = pool.acquire({8, 8});
    assert(pool.hits() == 1);
    assert(p2.data.data() == ptr1);  // same storage reused
    // reshape path: same numel, different shape reuses too
    pool.release(std::move(p2));
    llm::Tensor p3 = pool.acquire({4, 16});
    assert(pool.hits() == 2 && p3.shape[0] == 4 && p3.shape[1] == 16);
    // I85: 100 evict/reuse cycles over shared pool — misses stay bounded
    llm::KVCacheConfig pc;
    pc.paged = true; pc.page_size = 8;
    llm::KVCache kv(1, 64, 8, pc);
    kv.set_pool(&pool);
    size_t m_before = pool.misses();
    for (int s = 0; s < 100; ++s) {
        llm::Tensor k({1, 8}, 1.0f), v({1, 8}, 2.0f);
        for (int i = 0; i < 20; ++i) { kv.update(0, k, v); kv.advance(1); }
        kv.evict();
        kv.clear();
    }
    size_t m_after = pool.misses();
    std::cout << "pool misses before=" << m_before << " after=" << m_after
              << " hits=" << pool.hits() << "\n";
    assert(m_after - m_before <= 6);  // 3 pages x K/V after warmup, then all reuse
    assert(pool.hits() > 100);
    std::cout << "pool test passed\n";
    return 0;
}
