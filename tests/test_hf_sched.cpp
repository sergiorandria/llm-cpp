#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

#include "llm/config.h"
#include "llm/scheduler.h"
int main() {
    // D33: HF GPT-2 124M shape maps correctly
    {
        std::ofstream o("/tmp/hf_config.json");
        o << "{\"n_layer\":12,\"n_head\":12,\"n_embd\":768,\"n_positions\":1024,\"n_vocab\":50257}"
             "\n";
    }
    llm::Config c = llm::load_hf_config("/tmp/hf_config.json");
    assert(c.n_layers == 12 && c.n_heads == 12 && c.n_embd == 768);
    assert(c.block_size == 1024 && c.vocab_size == 50257);
    assert(llm::validate_config(c));
    // D34: warmup ramp + cosine to min + restart peaks
    llm::CosineScheduler s(0.1f, 10, 110, 0.01f, 1);
    assert(std::fabs(s.get_lr(0)) < 1e-6);
    assert(std::fabs(s.get_lr(10) - 0.1f) < 1e-5);
    float mid = s.get_lr(60), end = s.get_lr(110);
    assert(mid < 0.1f && mid > 0.01f);
    assert(std::fabs(end - 0.01f) < 1e-4);
    llm::CosineScheduler r(0.1f, 0, 100, 0.0f, 2);
    float peak0 = r.get_lr(0), peak1 = r.get_lr(50);
    assert(std::fabs(peak0 - 0.1f) < 1e-5 && std::fabs(peak1 - 0.1f) < 1e-5);  // restart peaks
    assert(r.get_lr(25) < 0.06f);                                              // cycle valley
    std::cout << "hf+sched test passed\n";
    return 0;
}
