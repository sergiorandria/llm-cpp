// G61/G62/G65/G70: server loopback tests (no side effects inside assert).
#include "llm/server.h"
#include <arpa/inet.h>
#include <cassert>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

static std::string http_post(int port, const std::string& path, const std::string& body) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { std::cerr << "socket failed\n"; exit(1); }
    struct sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (connect(fd, (struct sockaddr*)&a, sizeof a) != 0) { std::cerr << "connect failed\n"; exit(1); }
    std::string req = "POST " + path + " HTTP/1.1\r\nHost: x\r\nContent-Length: " +
                      std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    size_t sent = 0;
    while (sent < req.size()) {
        ssize_t w = send(fd, req.data() + sent, req.size() - sent, MSG_NOSIGNAL);
        if (w <= 0) { std::cerr << "send failed\n"; exit(1); }
        sent += (size_t)w;
    }
    std::string resp;
    char b[4096];
    ssize_t n;
    while ((n = recv(fd, b, sizeof b, 0)) > 0) resp.append(b, n);
    close(fd);
    return resp;
}
static std::string http_get(int port, const std::string& path) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { std::cerr << "socket failed\n"; exit(1); }
    struct sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (connect(fd, (struct sockaddr*)&a, sizeof a) != 0) { std::cerr << "connect failed\n"; exit(1); }
    std::string req = "GET " + path + " HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n";
    ssize_t w0 = send(fd, req.data(), req.size(), MSG_NOSIGNAL);
    if (w0 != (ssize_t)req.size()) { std::cerr << "send failed\n"; exit(1); }
    std::string resp;
    char b[4096];
    ssize_t n;
    while ((n = recv(fd, b, sizeof b, 0)) > 0) resp.append(b, n);
    close(fd);
    return resp;
}
static int status_of(const std::string& r) { return std::stoi(r.substr(9, 3)); }
static std::string body_of(const std::string& r) {
    size_t p = r.find("\r\n\r\n");
    return p == std::string::npos ? "" : r.substr(p + 4);
}

int main() {
    llm::Config cfg;
    cfg.vocab_size = 32; cfg.n_layers = 1; cfg.n_heads = 2; cfg.n_embd = 8; cfg.block_size = 16;
    llm::GPT model(cfg);
    llm::Tokenizer tok(256);
    llm::ServerConfig sc;
    sc.port = 0;  // ephemeral
    llm::Server srv(model, tok, sc);
    bool started = srv.start();
    assert(started);
    int port = srv.port();
    assert(port > 0);
    // G65 healthz
    std::string h = http_get(port, "/healthz");
    bool h200 = status_of(h) == 200;
    bool has_ok = body_of(h).find("\"status\":\"ok\"") != std::string::npos;
    assert(h200 && has_ok);
    // G61 completions == direct greedy decode
    std::string r = http_post(port, "/v1/completions", "{\"prompt\":\"Hi\",\"max_tokens\":4,\"temperature\":0}");
    bool r200 = status_of(r) == 200;
    assert(r200);
    std::string text = llm::json_get_string(body_of(r), "text", "#missing#");
    auto ids = tok.encode("Hi");
    auto gen = model.generate(ids, 4, 0.0f);
    std::string expect = tok.decode(std::vector<int>(gen.begin() + ids.size(), gen.end()));
    assert(text == expect);
    // 400 on bad temperature (calls hoisted)
    std::string rb = http_post(port, "/v1/completions", "{\"prompt\":\"Hi\",\"temperature\":-1}");
    bool is400 = status_of(rb) == 400;
    assert(is400);
    // G62 SSE shape
    std::string rs = http_post(port, "/v1/completions", "{\"prompt\":\"Hi\",\"max_tokens\":4,\"temperature\":0,\"stream\":true}");
    bool s200 = status_of(rs) == 200;
    std::string sb = body_of(rs);
    bool has_done = sb.find("data: [DONE]") != std::string::npos;
    bool has_ct = rs.find("text/event-stream") != std::string::npos;
    assert(s200 && has_done && has_ct);
    // G70 gate: max_concurrency=0 always 429
    llm::ServerConfig sc0;
    sc0.port = 0; sc0.max_concurrency = 0;
    llm::Server srv0(model, tok, sc0);
    bool st0 = srv0.start();
    assert(st0);
    std::string r429 = http_post(srv0.port(), "/v1/completions", "{\"prompt\":\"Hi\"}");
    bool got429 = status_of(r429) == 429;
    assert(got429);
    // dispatch-level 429 path parity: direct dispatch still 200 (gate is at connection layer)
    llm::HttpResponse d = srv.dispatch("POST", "/v1/completions", "{\"prompt\":\"Hi\",\"temperature\":0}");
    assert(d.status == 200);
    srv.stop();
    srv0.stop();
    std::cout << "server test passed port=" << port << "\n";
    return 0;
}
