#include "llm/logging.h"

#include <chrono>
#include <cmath>
#include <fstream>
namespace llm {
void log(LogLevel lvl, const std::string& msg) {
    const char* p = "INFO";
    if (lvl == LogLevel::DEBUG)
        p = "DEBUG";
    else if (lvl == LogLevel::WARN)
        p = "WARN";
    else if (lvl == LogLevel::ERROR)
        p = "ERROR";
    std::cout << "[" << p << "] " << msg << "\n";
}
std::string log_level_name(LogLevel lvl) {
    if (lvl == LogLevel::DEBUG) return "DEBUG";
    if (lvl == LogLevel::WARN) return "WARN";
    if (lvl == LogLevel::ERROR) return "ERROR";
    return "INFO";
}
uint64_t now_ms() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}
std::string format_json(LogLevel lvl, const std::string& msg, const std::string& fields) {
    std::string o = "{\"ts\":" + std::to_string(now_ms()) + ",\"level\":\"" + log_level_name(lvl) +
                    "\",\"msg\":\"";
    for (char c : msg) {
        if (c == '"')
            o += "\\\"";
        else if (c == '\\')
            o += "\\\\";
        else if (c == '\n')
            o += "\\n";
        else
            o += c;
    }
    o += "\"";
    if (!fields.empty()) o += "," + fields;
    o += "}";
    return o;
}
void log_json(LogLevel lvl, const std::string& msg, const std::string& fields) {
    std::cerr << format_json(lvl, msg, fields) << "\n";
}
MetricsLogger::MetricsLogger(const std::string& path) : path_(path) {}
void MetricsLogger::log_step(int step, float loss, float ppl, float lr, float grad_norm,
                             float tokens_sec) {
    std::ofstream out(path_, std::ios::app);
    out << "{\"step\":" << step << ",\"loss\":" << loss << ",\"ppl\":" << ppl << ",\"lr\":" << lr
        << ",\"grad_norm\":" << grad_norm << ",\"tokens_sec\":" << tokens_sec << "}\n";
}
}  // namespace llm
