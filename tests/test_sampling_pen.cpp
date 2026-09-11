#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "llm/sampling.h"
int main() {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 0.5f};
    // E46: freq penalty lowers repeated token exactly by freq*count + presence
    std::vector<int> gen = {2, 2, 2, 1};  // tok2 x3, tok1 x1
    auto out = llm::apply_freq_presence(logits, gen, 0.5f, 0.3f);
    assert(std::fabs(out[2] - (3.0f - 0.5f * 3 - 0.3f)) < 1e-6);
    assert(std::fabs(out[1] - (2.0f - 0.5f * 1 - 0.3f)) < 1e-6);
    assert(out[0] == logits[0] && out[3] == logits[3]);  // unseen untouched
    // E47: bias adds; -inf bans (never sampled over 100 draws)
    std::unordered_map<int, float> bias = {{0, 5.0f}, {3, -std::numeric_limits<float>::infinity()}};
    auto b = llm::apply_logit_bias(logits, bias);
    assert(b[0] == 8.0f);
    for (int i = 0; i < 100; ++i) {
        int t = llm::sample_temperature_seeded(b, 1.0f, 1000 + i);
        assert(t != 3);
    }
    // E44: stop on eos or stop id
    assert(llm::should_stop(1, 1, {}) == true);
    assert(llm::should_stop(5, 1, {5, 6}) == true);
    assert(llm::should_stop(4, 1, {5, 6}) == false);
    std::cout << "sampling penalties test passed\n";
    return 0;
}
