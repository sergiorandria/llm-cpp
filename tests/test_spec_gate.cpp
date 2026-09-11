// E48: speculative speedup gate — wall-clock spec vs greedy + equivalence.
// Asserts equivalence (never speed — timing is informational for the dashboard).
#include "llm/speculative.h"
#include "llm/model.h"
#include <cassert>
#include <chrono>
#include <iostream>
int main() {
    llm::Config ct;
    ct.vocab_size = 32; ct.n_embd = 16; ct.n_heads = 2; ct.n_layers = 4; ct.block_size = 64;
    ct.weight_tying = true;
    llm::Config cd = ct;
    cd.n_layers = 2;
    llm::GPT target(ct), draft(cd);
    llm::SpeculativeDecoder dec(draft, target, 4);
    std::vector<int> prompt = {1, 2, 3};
    auto t0 = std::chrono::steady_clock::now();
    auto spec = dec.generate(prompt, 12);
    auto t1 = std::chrono::steady_clock::now();
    auto greedy = target.generate(prompt, 12, 0.0f, 0);
    auto t2 = std::chrono::steady_clock::now();
    assert(spec == greedy);
    double ms_spec = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_greedy = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "spec_ms=" << ms_spec << " greedy_ms=" << ms_greedy
              << " speedup=" << (ms_greedy / (ms_spec + 1e-9)) << "\n";
    std::cout << "spec gate test passed\n";
    return 0;
}
