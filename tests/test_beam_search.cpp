#include "llm/beam.h"
#include "llm/model.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>

static float sequence_logprob(const llm::GPT& model, const std::vector<int>& prompt, const std::vector<int>& seq) {
    // seq is expected to be prompt + generated tokens
    // Compute cumulative logprob of generated part (tokens after prompt)
    if(seq.size() <= prompt.size()) return 0.0f;
    const auto& cfg = model.config();
    size_t vocab = cfg.vocab_size;
    size_t block_size = cfg.block_size;
    float total = 0.0f;
    // For each generated token position
    for(size_t pos = prompt.size(); pos < seq.size(); ++pos){
        // Context is seq[0..pos-1] truncated to block_size (as model does)
        size_t start = 0;
        if(pos > block_size) start = pos - block_size;
        std::vector<int> ctx(seq.begin()+start, seq.begin()+pos);
        if(ctx.empty()) continue;
        auto logits = model.forward(ctx);
        size_t T = logits.shape[0];
        std::vector<float> row(vocab);
        for(size_t j=0;j<vocab;++j) row[j]=logits(T-1,j);
        float maxv = row[0];
        for(size_t j=1;j<vocab;++j) maxv = std::max(maxv, row[j]);
        float sum=0;
        for(float v: row) sum += std::exp(v - maxv);
        float logsum = maxv + std::log(sum);
        int tok = seq[pos];
        // Clamp tok to vocab
        int v = ((tok % (int)vocab) + (int)vocab) % (int)vocab;
        float logp = row[v] - logsum;
        total += logp;
    }
    return total;
}

int main(){
    llm::Config cfg;
    cfg.vocab_size = 32;
    cfg.n_embd = 16;
    cfg.n_heads = 2;
    cfg.n_layers = 2;
    cfg.block_size = 32;
    cfg.weight_tying = true;
    llm::GPT model(cfg);

    std::vector<int> prompt = {1,2,3};
    size_t max_new = 8;
    size_t beam_width = 4;

    auto beam_out = llm::beam_search(model, prompt, max_new, beam_width);
    auto greedy_out = model.generate(prompt, max_new, 0.0f, 0);

    assert(beam_out.size() == prompt.size() + max_new);
    assert(greedy_out.size() == prompt.size() + max_new);
    // Beam search must return best scoring, not just beams[0]
    // Check that beam's cumulative logprob >= greedy's
    float beam_score = sequence_logprob(model, prompt, beam_out);
    float greedy_score = sequence_logprob(model, prompt, greedy_out);
    std::cout << "beam score " << beam_score << " greedy " << greedy_score << "\n";
    // Allow small epsilon for floating error; beam should be >= greedy
    assert(beam_score + 1e-5 >= greedy_score && "beam search should have score >= greedy");

    // Also test beam_width 1 equals greedy
    auto beam1 = llm::beam_search(model, prompt, max_new, 1);
    for(size_t i=0;i<beam1.size();++i) assert(beam1[i]==greedy_out[i]);
    std::cout << "beam_width=1 equals greedy passed\n";

    // Test that beam search does not always return beams[0] independent rollouts:
    // With beam_width 4, it should explore and find higher scoring path than naive independent rollouts would.
    // We already checked score >= greedy, which naive independent rollouts (old stub) would not guarantee.
    std::cout << "test_beam_search passed\n";
    return 0;
}
