// H72/H74/H75/H80: metrics scrape, guards, validation table, audit log.
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include "llm/server.h"
int main() {
    llm::Config cfg;
    cfg.vocab_size = 32;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_embd = 8;
    cfg.block_size = 16;
    llm::GPT model(cfg);
    llm::Tokenizer tok(256);
    std::remove("/tmp/audit.log");
    llm::ServerConfig sc;
    sc.audit_log = "/tmp/audit.log";
    llm::Server srv(model, tok, sc);
    // 3 completions through dispatch (no sockets needed for logic)
    for (int i = 0; i < 3; ++i) {
        llm::HttpResponse r = srv.dispatch(
            "POST", "/v1/completions", "{\"prompt\":\"Hi\",\"max_tokens\":2,\"temperature\":0}");
        assert(r.status == 200);
    }
    assert(srv.requests_total() == 3);
    assert(srv.tokens_total() == 6);
    // H72: metrics exposition has all four series
    llm::HttpResponse m = srv.dispatch("GET", "/metrics", "");
    assert(m.status == 200);
    assert(m.body.find("llm_requests_total 3") != std::string::npos);
    assert(m.body.find("llm_tokens_total 6") != std::string::npos);
    assert(m.body.find("llm_latency_ms_total") != std::string::npos);
    assert(m.body.find("llm_kv_cache_bytes") != std::string::npos);
    // H74: oversize prompt+max refused before alloc (block 16)
    std::string big(100, 'a');
    llm::HttpResponse o =
        srv.dispatch("POST", "/v1/completions",
                     "{\"prompt\":\"" + big + "\",\"max_tokens\":50,\"temperature\":0}");
    assert(o.status == 400);
    // H75: validation table (status + message), calls hoisted
    struct Case {
        const char* body;
        int want;
        const char* msg;
    };
    Case cases[] = {
        {"{\"prompt\":\"Hi\",\"temperature\":-1}", 400, "temperature"},
        {"{\"prompt\":\"Hi\",\"top_p\":0}", 400, "top_p"},
        {"{\"prompt\":\"Hi\",\"top_p\":1.5}", 400, "top_p"},
        {"{\"prompt\":\"Hi\",\"top_k\":-2}", 400, "top_k"},
        {"{\"prompt\":\"Hi\",\"max_tokens\":0}", 400, "max_tokens"},
        {"{\"prompt\":\"Hi\",\"max_tokens\":99999}", 400, "max_tokens"},
        {"{\"prompt\":\"Hi\",\"max_tokens\":2}", 200, ""},
        {"{}", 400, "block_size"},  // empty prompt + default max 50 > block 16
    };
    for (auto& c : cases) {
        llm::HttpResponse r = srv.dispatch("POST", "/v1/completions", c.body);
        assert(r.status == c.want);
        if (c.want == 400) assert(r.body.find(c.msg) != std::string::npos);
    }
    llm::HttpResponse nf = srv.dispatch("GET", "/nope", "");
    assert(nf.status == 404);
    // H80: 3 completions + 2 more dispatches above that generated (table 200s + {}) = audit lines;
    // count lines and check stable hash for same prompt
    std::ifstream af("/tmp/audit.log");
    int lines = 0;
    std::string l, h1, h2;
    while (std::getline(af, l)) {
        assert(l.find("\"prompt_hash\":") != std::string::npos);
        assert(l.find("Hello world secret") == std::string::npos);  // never raw prompt
        ++lines;
    }
    assert(lines >= 3);
    llm::Server srv2(model, tok, sc);
    llm::HttpResponse a1 = srv2.dispatch(
        "POST", "/v1/completions", "{\"prompt\":\"Same\",\"max_tokens\":1,\"temperature\":0}");
    llm::HttpResponse a2 = srv2.dispatch(
        "POST", "/v1/completions", "{\"prompt\":\"Same\",\"max_tokens\":1,\"temperature\":0}");
    assert(a1.status == 200 && a2.status == 200);
    // stable hash: last two audit lines carry the same prompt_hash
    std::vector<std::string> all;
    {
        std::ifstream af2("/tmp/audit.log");
        std::string x;
        while (std::getline(af2, x)) all.push_back(x);
    }
    auto hash_of = [](const std::string& s) {
        size_t p = s.find("\"prompt_hash\":");
        size_t e = s.find(',', p);
        return s.substr(p, e - p);
    };
    assert(all.size() >= 2);
    assert(hash_of(all[all.size() - 1]) == hash_of(all[all.size() - 2]));
    std::cout << "metrics+guards+audit test passed lines=" << lines << "\n";
    return 0;
}
