#include <cassert>
#include <iostream>

#include "llm/tokenizer.h"
int main() {
    auto p = llm::Tokenizer::split_pretokenize("Hello, world!");
    // ["Hello", ",", " world", "!"] — GPT-2 style: leading space attaches, comma splits
    assert(p.size() == 4);
    assert(p[0] == "Hello" && p[1] == "," && p[2] == " world" && p[3] == "!");
    auto q = llm::Tokenizer::split_pretokenize("I'd like 42 eggs");
    // ["I", "'d", " like", " 42"...] check contraction split + number
    assert(q.size() >= 4);
    assert(q[0] == "I" && q[1] == "'d");
    bool has_num = false;
    for (auto& t : q)
        if (t.find("42") != std::string::npos) has_num = true;
    assert(has_num);
    // concat == original
    std::string cat;
    for (auto& t : q) cat += t;
    assert(cat == "I'd like 42 eggs");
    std::cout << "pretok test passed\n";
    return 0;
}
