#include "llm/speculative.h"
namespace llm {
SpeculativeDecoder::SpeculativeDecoder(const GPT& draft, const GPT& target, size_t k)
    : draft_(draft), target_(target), k_(k) {}
std::vector<int> SpeculativeDecoder::generate(const std::vector<int>& prompt, size_t max_tokens) const {
    std::vector<int> out = prompt;
    for(size_t s=0; s<max_tokens; ){
        auto draft_tokens = draft_.generate(out, k_, 0.0f, 0);
        draft_tokens.erase(draft_tokens.begin(), draft_tokens.begin()+out.size());
        // Verify draft tokens with target in parallel (here sequential for stub)
        auto verify = target_.generate(out, draft_tokens.size(), 0.0f, 0);
        verify.erase(verify.begin(), verify.begin()+out.size());
        size_t accepted=0;
        for(size_t i=0;i<draft_tokens.size() && i<verify.size(); ++i){
            if(draft_tokens[i]==verify[i]) { out.push_back(draft_tokens[i]); accepted++; }
            else { out.push_back(verify[i]); accepted++; break; }
        }
        if(accepted==0) { auto t = target_.generate(out, 1, 0.0f, 0); out.push_back(t.back()); accepted=1; }
        s+=accepted;
    }
    return out;
}
}
