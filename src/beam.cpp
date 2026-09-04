#include "llm/beam.h"
namespace llm {
std::vector<int> beam_search(const GPT& model, const std::vector<int>& prompt, size_t max_new_tokens, size_t beam_width){
    if(beam_width<=1) return model.generate(prompt, max_new_tokens);
    // naive beam: keep beam_width candidates, expand greedy + sampling
    std::vector<std::vector<int>> beams(beam_width, prompt);
    for(size_t step=0; step<max_new_tokens; ++step){
        for(auto &b: beams) b = model.generate(b, 1, 1.0f, 0);
    }
    return beams[0];
}
}
