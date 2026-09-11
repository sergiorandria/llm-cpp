#pragma once
#include <cstdint>
#include <iostream>
#include <string>
namespace llm {
enum class LogLevel { DEBUG, INFO, WARN, ERROR };
void log(LogLevel lvl, const std::string& msg);
#define LOG_INFO(msg) llm::log(llm::LogLevel::INFO, msg)
#define LOG_WARN(msg) llm::log(llm::LogLevel::WARN, msg)
// H71: structured JSON logging. format_json is pure (testable); log_json prints to stderr.
std::string log_level_name(LogLevel lvl);
uint64_t now_ms();
std::string format_json(LogLevel lvl, const std::string& msg, const std::string& fields = "");
void log_json(LogLevel lvl, const std::string& msg, const std::string& fields = "");
// D37: per-step JSONL metrics logger (train.log.jsonl)
class MetricsLogger {
   public:
    explicit MetricsLogger(const std::string& path);
    void log_step(int step, float loss, float ppl, float lr, float grad_norm, float tokens_sec);

   private:
    std::string path_;
};
}  // namespace llm
