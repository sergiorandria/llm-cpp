#include <cassert>
#include <fstream>
#include <iostream>

#include "llm/tokenizer.h"
int main() {
    llm::Tokenizer tok(512);
    // ASCII
    std::string s1 = "Hello, world!";
    assert(tok.decode(tok.encode(s1)) == s1);
    // Malagasy with accents + curly quotes (multi-byte UTF-8)
    std::string s2 = "Ny vetso — “aza gaga raha hamaly aho”";
    auto ids2 = tok.encode(s2);
    std::string r2 = tok.decode(ids2);
    assert(r2 == s2);
    // Emoji (4-byte sequences)
    std::string s3 = "hello 🌍🌍 world 🎉";
    assert(tok.decode(tok.encode(s3)) == s3);
    // Invalid UTF-8 bytes passthrough (byte-level never throws)
    std::string s4("\xff\xfe\x00A", 4);
    assert(tok.decode(tok.encode(s4)) == s4);
    // Real corpus file byte-exact (first 64KB)
    std::ifstream in("data/malagasy/processed/input.txt", std::ios::binary);
    if (in) {
        std::string txt((std::istreambuf_iterator<char>(in)), {});
        txt = txt.substr(0, 65536);
        assert(tok.decode(tok.encode(txt)) == txt);
        std::cout << "corpus 64KB roundtrip ok bytes=" << txt.size() << "\n";
    } else {
        std::cout << "corpus file missing, skipped\n";
    }
    std::cout << "utf8 test passed\n";
    return 0;
}
