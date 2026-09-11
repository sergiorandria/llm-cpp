#include <cassert>
#include <iostream>

#include "llm/logging.h"
#include "llm/model.h"
#include "llm/profiling.h"
int main() {
    // H71: JSON line parses (ts/level/msg keys, brace balance)
    std::string line = llm::format_json(llm::LogLevel::INFO, "hello", "\"step\":3");
    assert(!line.empty() && line.front() == '{' && line.back() == '}');
    assert(line.find("\"level\":\"INFO\"") != std::string::npos);
    assert(line.find("\"msg\":\"hello\"") != std::string::npos);
    assert(line.find("\"ts\":") != std::string::npos);
    assert(line.find("\"step\":3") != std::string::npos);
    assert(llm::format_json(llm::LogLevel::WARN, "x").find("\"level\":\"WARN\"") !=
           std::string::npos);
    // H73: timers record after a forward
    llm::Config cfg;
    cfg.vocab_size = 16;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_embd = 8;
    cfg.block_size = 8;
    llm::GPT m(cfg);
    (void)m.forward({1, 2, 3, 4});
    std::string rep = llm::profiling_report();
    std::cout << rep;
    assert(rep.find("forward") != std::string::npos);
    assert(rep.find("attn") != std::string::npos);
    assert(rep.find("block") != std::string::npos);
    std::cout << "log+prof test passed\n";
    return 0;
}
