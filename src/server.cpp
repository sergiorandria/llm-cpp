#include "llm/server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include "llm/version.h"

namespace llm {

std::string json_escape(const std::string& s) {
    // Guarantees valid UTF-8 JSON: valid multi-byte sequences pass through,
    // stray bytes become U+FFFD (byte-level tokens can emit lone high bytes).
    std::string o;
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = s[i];
        if (c == '"') {
            o += "\\\"";
            ++i;
        } else if (c == '\\') {
            o += "\\\\";
            ++i;
        } else if (c == '\n') {
            o += "\\n";
            ++i;
        } else if (c == '\t') {
            o += "\\t";
            ++i;
        } else if (c < 0x20) {
            char b[7];
            snprintf(b, sizeof b, "\\u%04x", c);
            o += b;
            ++i;
        } else if (c < 0x80) {
            o += (char)c;
            ++i;
        } else {
            size_t need = 0;
            if ((c & 0xE0) == 0xC0)
                need = 2;
            else if ((c & 0xF0) == 0xE0)
                need = 3;
            else if ((c & 0xF8) == 0xF0)
                need = 4;
            bool ok = need > 0 && i + need <= n;
            for (size_t k = 1; ok && k < need; ++k)
                if (((unsigned char)s[i + k] & 0xC0) != 0x80) ok = false;
            if (ok) {
                o.append(s, i, need);
                i += need;
            } else {
                o += "\xEF\xBF\xBD";
                ++i;
            }  // U+FFFD
        }
    }
    return o;
}

static std::string extract_raw(const std::string& body, const std::string& key) {
    std::string q = "\"" + key + "\"";
    size_t p = body.find(q);
    if (p == std::string::npos) return "";
    p = body.find(':', p + q.size());
    if (p == std::string::npos) return "";
    size_t s = body.find_first_not_of(" \t", p + 1);
    if (s == std::string::npos) return "";
    if (body[s] == '"') {  // string (no \u handling needed for prompts; \" and \\ handled)
        std::string o;
        for (size_t i = s + 1; i < body.size(); ++i) {
            if (body[i] == '\\' && i + 1 < body.size()) {
                char e = body[++i];
                if (e == 'n')
                    o += '\n';
                else if (e == 't')
                    o += '\t';
                else if (e == '"')
                    o += '"';
                else if (e == '\\')
                    o += '\\';
                else {
                    o += '\\';
                    o += e;
                }
            } else if (body[i] == '"')
                break;
            else
                o += body[i];
        }
        return std::string("\x01") + o;  // marker: string value
    }
    size_t e = body.find_first_of(",}]", s);
    return body.substr(s, e - s);
}

std::string json_get_string(const std::string& body, const std::string& key,
                            const std::string& def) {
    std::string r = extract_raw(body, key);
    if (r.empty() || r[0] != '\x01') return def;
    return r.substr(1);
}
double json_get_number(const std::string& body, const std::string& key, double def) {
    std::string r = extract_raw(body, key);
    if (r.empty() || r[0] == '\x01') return def;
    try {
        return std::stod(r);
    } catch (...) {
        return def;
    }
}
bool json_get_bool(const std::string& body, const std::string& key, bool def) {
    std::string r = extract_raw(body, key);
    if (r == "true") return true;
    if (r == "false") return false;
    return def;
}

Server::Server(GPT& model, Tokenizer& tok, ServerConfig cfg)
    : model_(model), tok_(tok), cfg_(cfg) {}

bool Server::try_enter() {
    int cur = in_flight_.load();
    while ((size_t)cur < cfg_.max_concurrency) {
        if (in_flight_.compare_exchange_weak(cur, cur + 1)) return true;
    }
    return false;
}
void Server::leave() {
    in_flight_--;
}

static HttpResponse err400(const std::string& msg) {
    return {400, "application/json", "{\"error\":\"" + json_escape(msg) + "\"}"};
}

void Server::audit_append(const std::string& prompt, size_t new_tokens) {
    if (cfg_.audit_log.empty()) return;
    // H80: hash only — never raw prompt bytes (PII-safe default)
    size_t h = std::hash<std::string>{}(prompt);
    std::ostringstream o;
    o << "{\"ts\":"
      << std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
             .count()
      << ",\"prompt_hash\":" << h << ",\"new_tokens\":" << new_tokens
      << ",\"params\":" << model_.num_parameters() << "}\n";
    std::ofstream f(cfg_.audit_log, std::ios::app);
    f << o.str();
}

HttpResponse Server::serve_completions(const std::string& body, bool stream) {
    auto t0 = std::chrono::steady_clock::now();
    std::string prompt = json_get_string(body, "prompt", "");
    double max_t = json_get_number(body, "max_tokens", (double)cfg_.max_tokens_default);
    double temp = json_get_number(body, "temperature", 1.0);
    double top_p = json_get_number(body, "top_p", 1.0);
    double top_k = json_get_number(body, "top_k", 0);
    if (temp < 0) return err400("temperature must be >= 0");
    if (top_p <= 0 || top_p > 1) return err400("top_p must be in (0, 1]");
    if (top_k < 0) return err400("top_k must be >= 0");
    if (max_t <= 0 || max_t > 4096) return err400("max_tokens must be in [1, 4096]");
    auto ids = tok_.encode(prompt);
    // H74: refuse oversize before any big alloc
    if (ids.size() + (size_t)max_t > model_.config().block_size)
        return err400("prompt + max_tokens exceeds block_size");
    std::vector<int> out;
    if (temp == 0)
        out = model_.generate(ids, (size_t)max_t, 0.0f, 0);
    else {
        auto go = model_.generate_with_logprobs(ids, (size_t)max_t, (float)temp, (int)top_k,
                                                (float)top_p);
        out = go.tokens;
    }
    size_t new_tokens = out.size() > ids.size() ? out.size() - ids.size() : 0;
    requests_total_++;
    tokens_total_ += new_tokens;
    latency_ms_total_ += (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
    audit_append(prompt, new_tokens);
    std::vector<int> completion(out.begin() + ids.size(), out.end());
    std::string text = tok_.decode(completion);
    if (!stream) {
        std::ostringstream o;
        o << "{\"id\":\"cmpl-1\",\"object\":\"text_completion\",\"choices\":[{\"text\":\""
          << json_escape(text) << "\",\"index\":0,\"finish_reason\":\"length\"}]}";
        return {200, "application/json", o.str()};
    }
    // SSE: one event per completion token + [DONE]
    std::ostringstream o;
    std::vector<int> prefix = ids;
    for (int t : completion) {
        std::vector<int> one = {t};
        o << "data: {\"choices\":[{\"text\":\"" << json_escape(tok_.decode(one)) << "\"}]}\n\n";
        prefix.push_back(t);
    }
    o << "data: [DONE]\n\n";
    return {200, "text/event-stream", o.str()};
}

HttpResponse Server::serve_chat(const std::string& body, bool stream) {
    // Take last {"content": "..."} as the prompt (minimal chat support)
    std::string prompt;
    size_t pos = 0;
    while (true) {
        size_t p = body.find("\"content\"", pos);
        if (p == std::string::npos) break;
        p = body.find(':', p);
        size_t s = body.find('"', p);
        std::string acc;
        for (size_t i = s + 1; i < body.size(); ++i) {
            if (body[i] == '\\' && i + 1 < body.size()) {
                acc += body[++i];
            } else if (body[i] == '"')
                break;
            else
                acc += body[i];
        }
        prompt = acc;
        pos = s + 1;
    }
    // Reuse completions path with prompt substituted
    std::ostringstream nb;
    nb << "{\"prompt\":\"" << json_escape(prompt) << "\"";
    for (auto k : {"max_tokens", "temperature", "top_p", "top_k"}) {
        std::string r = extract_raw(body, k);
        if (!r.empty() && r[0] != '\x01') nb << ",\"" << k << "\":" << r;
    }
    if (stream) nb << ",\"stream\":true";
    nb << "}";
    HttpResponse r = serve_completions(nb.str(), stream);
    if (!stream && r.status == 200) {
        // reshape to chat shape: extract text
        std::string text = json_get_string(r.body, "text", "");
        std::ostringstream o;
        o << "{\"id\":\"chat-1\",\"object\":\"chat.completion\",\"choices\":[{\"message\":{"
             "\"role\":\"assistant\",\"content\":\""
          << json_escape(text) << "\"},\"finish_reason\":\"length\"}]}";
        r.body = o.str();
    }
    return r;
}

HttpResponse Server::dispatch(const std::string& method, const std::string& path,
                              const std::string& body) {
    if (method == "GET" && path == "/healthz") {
        std::ostringstream o;
        o << "{\"status\":\"ok\",\"version\":\"" << LLM_CPP_VERSION
          << "\",\"params\":" << model_.num_parameters() << "}";
        return {200, "application/json", o.str()};
    }
    if (method == "GET" && path == "/metrics") {
        // H72: Prometheus exposition
        const auto& c = model_.config();
        std::ostringstream o;
        o << "# HELP llm_requests_total Total completion requests\n# TYPE llm_requests_total "
             "counter\n"
          << "llm_requests_total " << requests_total_.load() << "\n"
          << "# HELP llm_tokens_total Total generated tokens\n# TYPE llm_tokens_total counter\n"
          << "llm_tokens_total " << tokens_total_.load() << "\n"
          << "# HELP llm_latency_ms_total Total generate latency ms\n# TYPE llm_latency_ms_total "
             "counter\n"
          << "llm_latency_ms_total " << latency_ms_total_.load() << "\n"
          << "# HELP llm_kv_cache_bytes Reserved KV-cache bytes (contiguous layout)\n# TYPE "
             "llm_kv_cache_bytes gauge\n"
          << "llm_kv_cache_bytes " << (uint64_t)c.n_layers * c.block_size * c.n_embd * 4 * 2
          << "\n";
        return {200, "text/plain; version=0.0.4", o.str()};
    }
    if (method == "POST" && path == "/v1/completions")
        return serve_completions(body, json_get_bool(body, "stream", false));
    if (method == "POST" && path == "/v1/chat/completions")
        return serve_chat(body, json_get_bool(body, "stream", false));
    return {404, "application/json", "{\"error\":\"not found\"}"};
}

static std::string reason(int s) {
    if (s == 200) return "OK";
    if (s == 400) return "Bad Request";
    if (s == 404) return "Not Found";
    if (s == 429) return "Too Many Requests";
    return "Error";
}

void Server::serve_connection(int fd) {
    // Read headers + body (Content-Length)
    std::string req;
    char buf[4096];
    ssize_t n;
    size_t header_end = std::string::npos;
    while ((n = recv(fd, buf, sizeof buf, 0)) > 0) {
        req.append(buf, n);
        header_end = req.find("\r\n\r\n");
        if (header_end != std::string::npos) break;
        if (req.size() > 1 << 20) break;
    }
    if (header_end == std::string::npos) {
        close(fd);
        return;
    }
    std::istringstream hs(req.substr(0, header_end));
    std::string method, path, ver;
    hs >> method >> path >> ver;
    size_t cl = 0;
    {
        size_t p = req.find("Content-Length:");
        if (p != std::string::npos && p < header_end) cl = std::stoul(req.substr(p + 15));
    }
    std::string body = req.substr(header_end + 4);
    while (body.size() < cl && (n = recv(fd, buf, sizeof buf, 0)) > 0) body.append(buf, n);
    if (body.size() > cl) body.resize(cl);
    HttpResponse r;
    if (!try_enter())
        r = {429, "application/json", "{\"error\":\"overloaded\"}"};
    else {
        r = dispatch(method, path, body);
        leave();
    }
    std::ostringstream o;
    o << "HTTP/1.1 " << r.status << " " << reason(r.status) << "\r\n"
      << "Content-Type: " << r.content_type << "\r\n"
      << "Content-Length: " << r.body.size() << "\r\n"
      << "Connection: close\r\n\r\n"
      << r.body;
    std::string s = o.str();
    size_t sent = 0;
    while (sent < s.size()) {
        ssize_t w = send(fd, s.data() + sent, s.size() - sent, MSG_NOSIGNAL);
        if (w <= 0) break;
        sent += (size_t)w;
    }
    close(fd);
}

void Server::accept_loop() {
    while (running_.load()) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(listen_fd_, &rfds);
        struct timeval tv{0, 200000};  // 200ms so stop() is noticed promptly
        int rc = select(listen_fd_ + 1, &rfds, nullptr, nullptr, &tv);
        if (rc <= 0) continue;
        int fd = accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) continue;
        std::thread(&Server::serve_connection, this, fd).detach();  // G70: bounded by gate
    }
}

bool Server::start() {
    std::signal(SIGPIPE, SIG_IGN);  // never die on closed peer; send() returns EPIPE
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) return false;
    int one = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)cfg_.port);
    if (bind(listen_fd_, (struct sockaddr*)&addr, sizeof addr) != 0) {
        close(listen_fd_);
        return false;
    }
    socklen_t len = sizeof addr;
    getsockname(listen_fd_, (struct sockaddr*)&addr, &len);
    bound_port_ = ntohs(addr.sin_port);
    if (listen(listen_fd_, 64) != 0) {
        close(listen_fd_);
        return false;
    }
    running_ = true;
    accept_thr_ = std::thread(&Server::accept_loop, this);
    return true;
}

void Server::stop() {
    bool was = running_.exchange(false);
    if (listen_fd_ >= 0) {
        shutdown(listen_fd_, SHUT_RDWR);
        close(listen_fd_);
        listen_fd_ = -1;
    }
    if (was && accept_thr_.joinable()) accept_thr_.join();
}

}  // namespace llm
