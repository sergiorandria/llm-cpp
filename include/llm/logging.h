#pragma once
#include <string>
#include <iostream>
namespace llm {
enum class LogLevel { DEBUG, INFO, WARN, ERROR };
void log(LogLevel lvl, const std::string& msg);
#define LOG_INFO(msg) llm::log(llm::LogLevel::INFO, msg)
#define LOG_WARN(msg) llm::log(llm::LogLevel::WARN, msg)
}
