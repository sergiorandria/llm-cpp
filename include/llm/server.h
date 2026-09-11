#pragma once
#include "model.h"
#include "tokenizer.h"
#include <atomic>
#include <functional>
#include <string>
#include <thread>
namespace llm {
// G61/G62/G65/G70: minimal OpenAI-compatible HTTP server (POSIX sockets, no deps).
struct ServerConfig {
    int port = 8080;             // 0 = ephemeral (tests)
    size_t max_concurrency = 8;  // 0 = always 429 (tests); excess -> 429
    size_t max_tokens_default = 50;
};
struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
};
// Tiny JSON helpers (flat request objects only)
std::string json_escape(const std::string& s);
std::string json_get_string(const std::string& body, const std::string& key, const std::string& def = "");
double json_get_number(const std::string& body, const std::string& key, double def);
bool json_get_bool(const std::string& body, const std::string& key, bool def);
class Server {
public:
    Server(GPT& model, Tokenizer& tok, ServerConfig cfg = {});
    ~Server() { stop(); }
    // Pure request router (no sockets) — unit testable
    HttpResponse dispatch(const std::string& method, const std::string& path, const std::string& body);
    // Networking
    bool start();   // bind+listen+accept thread
    void stop();    // stop accepting, drain in-flight, join
    int port() const { return bound_port_.load(); }
    // G70: concurrency gate (public for tests)
    bool try_enter();
    void leave();
private:
    HttpResponse serve_completions(const std::string& body, bool stream);
    HttpResponse serve_chat(const std::string& body, bool stream);
    void accept_loop();
    void serve_connection(int fd);
    GPT& model_;
    Tokenizer& tok_;
    ServerConfig cfg_;
    std::atomic<bool> running_{false};
    std::atomic<int> bound_port_{-1};
    std::atomic<int> in_flight_{0};
    int listen_fd_ = -1;
    std::thread accept_thr_;
};
} // namespace llm
