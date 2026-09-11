// J91: generated version is non-empty and well-formed.
#include "llm/version.h"
#include <cassert>
#include <iostream>
#include <string>
int main() {
    std::string v = LLM_CPP_VERSION;
    assert(!v.empty());
    assert(v.find('.') != std::string::npos || v.find('+') != std::string::npos);
    assert(LLM_CPP_VERSION_MAJOR >= 0 && LLM_CPP_VERSION_MINOR >= 0 && LLM_CPP_VERSION_PATCH >= 0);
    std::cout << "version test passed: " << v << "\n";
    return 0;
}
