#include "llm/beam.h"
namespace llm {
std::vector<int> beam_search(const GPT& model, const std::vector<int>& prompt, size_t max_new_tokens, size_t beam_width){
    (void)beam_width;
    // stub: fallback to greedy
    return model.generate(prompt, max_new_tokens);
}
}
