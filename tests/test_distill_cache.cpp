#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/distill.h"
int main() {
    llm::Config ct, cs;
    ct.vocab_size = 16;
    ct.n_layers = 2;
    ct.n_heads = 2;
    ct.n_embd = 8;
    ct.block_size = 8;
    cs = ct;
    cs.n_layers = 1;
    llm::GPT teacher(ct), student(cs);
    std::vector<std::vector<int>> batches = {{1, 2, 3, 4}, {5, 6, 7, 8}, {2, 4, 6, 8}};
    // calls hoisted (NDEBUG elides assert args)
    llm::cache_teacher_logits(teacher, batches, "/tmp/teacher.bin");
    auto cached = llm::load_cached_logits("/tmp/teacher.bin");
    assert(cached.size() == batches.size());
    // cached logits bit-match live
    for (size_t i = 0; i < batches.size(); ++i) {
        auto live = teacher.forward(batches[i]);
        assert(live.shape == cached[i].shape);
        float md = 0;
        for (size_t j = 0; j < live.data.size(); ++j)
            md = std::max(md, std::fabs(live.data[j] - cached[i].data[j]));
        assert(md == 0.0f);
    }
    // cached KL grads match live within 1e-6
    for (size_t i = 0; i < batches.size(); ++i) {
        llm::GPT s1(cs), s2(cs);
        for (size_t k = 0; k < s1.parameters().size(); ++k)
            s2.parameters()[k]->data = s1.parameters()[k]->data;
        llm::distill_step(s1, teacher, batches[i]);
        llm::distill_step_cached(s2, cached[i], batches[i]);
        auto g1 = s1.parameters(), g2 = s2.parameters();
        float md = 0;
        for (size_t k = 0; k < g1.size(); ++k)
            for (size_t j = 0; j < g1[k]->grad.size(); ++j)
                md = std::max(md, std::fabs(g1[k]->grad[j] - g2[k]->grad[j]));
        std::cout << "batch " << i << " grad maxd=" << md << "\n";
        assert(md < 1e-6f);
    }
    std::cout << "distill cache test passed\n";
    return 0;
}
