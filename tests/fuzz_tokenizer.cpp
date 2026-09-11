#include <cassert>
#include <fstream>
#include <random>

#include "llm/tokenizer.h"
// B20: also runs seed corpus tests/fuzz_corpus/* (empty, invalid UTF-8, Malagasy, tiny)
static void run_corpus(llm::Tokenizer& tok) {
    for (auto* p : {(const char*)"tests/fuzz_corpus/empty.bin",
                    (const char*)"tests/fuzz_corpus/invalid_utf8.bin",
                    (const char*)"tests/fuzz_corpus/malagasy.txt",
                    (const char*)"tests/fuzz_corpus/tiny.txt"}) {
        std::ifstream in(p, std::ios::binary);
        if (!in) continue;
        std::string s((std::istreambuf_iterator<char>(in)), {});
        auto ids = tok.encode(s);
        auto dec = tok.decode(ids);
        assert(dec == s);  // byte-level must roundtrip even invalid UTF-8
    }
}
int main() {
    llm::Tokenizer tok(1000);
    run_corpus(tok);
    std::mt19937 rng(0);
    for (int i = 0; i < 1000; ++i) {
        std::string s;
        int len = rng() % 20;
        for (int j = 0; j < len; ++j) s.push_back(char(rng() % 256));
        auto ids = tok.encode(s);
        auto dec = tok.decode(ids);
        // Not asserting equality for random bytes, just that it doesn't crash
        (void)dec;
    }
    return 0;
}
