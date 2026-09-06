#include "llm/beam.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace llm {

struct BeamState {
    std::vector<int> tokens;
    float score = -1e30f; // cumulative log prob
};

std::vector<int> beam_search(const GPT& model, const std::vector<int>& prompt, size_t max_new_tokens, size_t beam_width){
    if(beam_width<=1) return model.generate(prompt, max_new_tokens, 0.0f, 0);
    if(max_new_tokens==0) return prompt;
    const auto& cfg = model.config();
    size_t vocab = cfg.vocab_size;
    size_t block_size = cfg.block_size;

    std::vector<BeamState> beams;
    beams.push_back({prompt, 0.0f});

    for(size_t step=0; step<max_new_tokens; ++step){
        std::vector<BeamState> candidates;
        candidates.reserve(beams.size()*beam_width);
        for(const auto& beam : beams){
            // Context for forward: last block_size tokens (like generate does)
            std::vector<int> ctx = beam.tokens;
            if(ctx.size() > block_size){
                ctx.assign(beam.tokens.end() - block_size, beam.tokens.end());
            }
            if(ctx.empty()){
                // No context — cannot forward; fallback to greedy single step
                // Use model.generate for one token
                auto out = model.generate(beam.tokens, 1, 0.0f, 0);
                int tok = out.back();
                // Assign score as 0 for this fallback (no logprob)
                candidates.push_back({{beam.tokens}, beam.score});
                candidates.back().tokens.push_back(tok);
                continue;
            }
            Tensor logits = model.forward(ctx);
            size_t T = logits.shape[0];
            // last row
            std::vector<float> row(vocab);
            for(size_t j=0;j<vocab;++j) row[j]=logits(T-1,j);
            float maxv = row[0];
            for(size_t j=1;j<vocab;++j) maxv = std::max(maxv, row[j]);
            float sum = 0;
            for(float v: row) sum += std::exp(v - maxv);
            float logsum = maxv + std::log(sum);
            // Get top beam_width indices by logit (same order as logprob)
            std::vector<int> idx(vocab);
            std::iota(idx.begin(), idx.end(), 0);
            size_t k = std::min(beam_width, vocab);
            std::partial_sort(idx.begin(), idx.begin()+k, idx.end(),
                [&](int a, int b){ return row[a] > row[b]; });
            for(size_t j=0;j<k;++j){
                int tok = idx[j];
                float logp = row[tok] - logsum;
                BeamState nb;
                nb.tokens = beam.tokens;
                nb.tokens.push_back(tok);
                nb.score = beam.score + logp;
                candidates.push_back(std::move(nb));
            }
        }
        // Keep top beam_width candidates by score
        // If candidates fewer than beam_width, keep all
        std::sort(candidates.begin(), candidates.end(),
            [](const BeamState& a, const BeamState& b){ return a.score > b.score; });
        if(candidates.size() > beam_width) candidates.resize(beam_width);
        beams = std::move(candidates);
        if(beams.empty()) break;
    }
    // Return best scoring beam
    if(beams.empty()) return prompt;
    auto best = std::max_element(beams.begin(), beams.end(),
        [](const BeamState& a, const BeamState& b){ return a.score < b.score; });
    return best->tokens;
}

}
