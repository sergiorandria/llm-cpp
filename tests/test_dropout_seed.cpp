#include "llm/sampling.h"
#include "llm/tensor.h"
#include "llm/utils.h"
#include <cassert>
#include <iostream>
#include <random>
int main() {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 0.5f, -1.0f, 0.0f, 1.5f, 2.5f};
    // Seeded sampling deterministic
    int a = llm::sample_temperature_seeded(logits, 1.0f, 123);
    int b = llm::sample_temperature_seeded(logits, 1.0f, 123);
    assert(a == b);
    int c = llm::sample_top_k_seeded(logits, 3, 1.0f, 7);
    int d = llm::sample_top_k_seeded(logits, 3, 1.0f, 7);
    assert(c == d);
    int e = llm::sample_top_p_seeded(logits, 0.9f, 1.0f, 99);
    int f = llm::sample_top_p_seeded(logits, 0.9f, 1.0f, 99);
    assert(e == f);
    // global seed threads through unseeded API
    llm::set_global_seed(555);
    int g1 = llm::sample_temperature(logits, 1.0f);
    llm::set_global_seed(555);
    int g2 = llm::sample_temperature(logits, 1.0f);
    assert(g1 == g2);
    // dropout determinism: same mt19937 seed => identical mask
    llm::Tensor x({2, 8}, 1.0f);
    std::mt19937 r1(42), r2(42);
    auto y1 = x.dropout(0.5f, r1);
    auto y2 = x.dropout(0.5f, r2);
    assert(y1.data == y2.data);
    std::mt19937 r3(43);
    auto y3 = x.dropout(0.5f, r3);
    assert(y3.data != y1.data);
    std::cout << "dropout determinism test passed\n";
    return 0;
}
