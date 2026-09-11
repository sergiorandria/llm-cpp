#include "llm/tokenizer.h"
#include <cassert>
#include <iostream>
int main() {
    llm::Tokenizer tok(512);
    // emoji split across two chunks: full decode must equal streamed concat + empty carry
    std::string s = "hi 🌍 ok 🎉!";
    auto ids = tok.encode(s);
    size_t half = ids.size() / 2;
    std::vector<int> c1(ids.begin(), ids.begin() + half), c2(ids.begin() + half, ids.end());
    std::string carry;
    std::string o1 = tok.decode_incremental(c1, carry);
    std::string o2 = tok.decode_incremental(c2, carry);
    std::string tail = carry; carry.clear();
    assert(o1 + o2 + tail == s);
    assert(carry.empty());
    // single-byte stream never holds back
    std::string c;
    assert(tok.decode_incremental(tok.encode("abc"), c) == "abc" && c.empty());
    std::cout << "streaming test passed\n";
    return 0;
}
