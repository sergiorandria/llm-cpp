#pragma once
#include <string>
#include <iostream>
namespace llm {
enum class LogLevel { DEBUG, INFO, WARN, ERROR };
void log(LogLevel lvl, const std::string& msg);
#define LOG_INFO(msg) llm::log(llm::LogLevel::INFO, msg)
#define LOG_WARN(msg) llm::log(llm::LogLevel::WARN, msg)
// D37: per-step JSONL metrics logger (train.log.jsonl)
class MetricsLogger {
public:
    explicit MetricsLogger(const std::string& path);
    void log_step(int step, float loss, float ppl, float lr, float grad_norm, float tokens_sec);
private:
    std::string path_;
};
}
