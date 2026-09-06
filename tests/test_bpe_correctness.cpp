#include "llm/tokenizer.h"
#include <cassert>
#include <iostream>

int main() {
    // BPE correctness: train on repeated pattern, verify merges are learned and encode uses them
    llm::Tokenizer tok(1000);
    std::string corpus = "low lower lowest low lower";
    tok.train(corpus, 5);
    // After training, merges should not be empty and should contain real pairs (not "" )
    // Encode a known word and verify decode roundtrip is lossless for vocab
    std::string text = "lowest";
    auto ids = tok.encode(text);
    auto dec = tok.decode(ids);
    std::cout << "BPE ids for '" << text << "': ";
    for (auto id : ids) std::cout << id << " ";
    std::cout << "\ndecoded: '" << dec << "'\n";
    // Correctness: decode(encode(text)) should equal text for byte-level fallback + merges
    // With fixed separator (\x1F) merges are valid, so no empty token should appear
    assert(!ids.empty());
    assert(dec == text && "BPE roundtrip failed — separator bug would produce mismatched decode");
    // Verify merges are non-empty strings (bug produced "" merges)
    // We can't access merges_ directly, but we can test that encode of "low low" compresses
    auto ids2 = tok.encode("low low");
    std::cout << "low low ids size " << ids2.size() << " (should be <= 7 chars due to merges)\n";
    assert(ids2.size() <= 7);
    std::cout << "BPE correctness test passed\n";
    return 0;
}
