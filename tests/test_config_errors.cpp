#include <cassert>
#include <iostream>

#include "llm/config.h"
int main() {
    llm::Config ok;
    std::string err;
    bool vok = llm::validate_config_verbose(ok, err);
    assert(vok && err.empty());
    int n = 0;
    {
        llm::Config c;
        c.n_embd = 10;
        c.n_heads = 3;
        bool r = llm::validate_config_verbose(c, err);
        assert(!r && err.find("n_heads") != std::string::npos);
        ++n;
    }
    {
        llm::Config c;
        c.vocab_size = 0;
        bool r = llm::validate_config_verbose(c, err);
        assert(!r && err.find("vocab_size") != std::string::npos);
        ++n;
    }
    {
        llm::Config c;
        c.n_layers = 0;
        bool r = llm::validate_config_verbose(c, err);
        assert(!r && err.find("n_layers") != std::string::npos);
        ++n;
    }
    {
        llm::Config c;
        c.use_alibi = true;
        c.pos_encoding = llm::PosEncoding::RoPE;
        bool r = llm::validate_config_verbose(c, err);
        assert(!r && err.find("mutually exclusive") != std::string::npos);
        ++n;
    }
    {
        llm::Config c;
        c.rope_scaling = 0.5f;
        bool r = llm::validate_config_verbose(c, err);
        assert(!r && err.find("rope_scaling") != std::string::npos);
        ++n;
    }
    std::cout << "config errors test passed (" << n << " bad configs)\n";
    return 0;
}
