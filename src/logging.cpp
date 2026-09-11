#include "llm/logging.h"
#include <cmath>
#include <fstream>
namespace llm {
void log(LogLevel lvl, const std::string& msg){
    const char* p="INFO";
    if(lvl==LogLevel::DEBUG) p="DEBUG";
    else if(lvl==LogLevel::WARN) p="WARN";
    else if(lvl==LogLevel::ERROR) p="ERROR";
    std::cout<<"["<<p<<"] "<<msg<<"\n";
}
MetricsLogger::MetricsLogger(const std::string& path) : path_(path) {}
void MetricsLogger::log_step(int step, float loss, float ppl, float lr, float grad_norm, float tokens_sec) {
    std::ofstream out(path_, std::ios::app);
    out << "{\"step\":" << step << ",\"loss\":" << loss << ",\"ppl\":" << ppl
        << ",\"lr\":" << lr << ",\"grad_norm\":" << grad_norm
        << ",\"tokens_sec\":" << tokens_sec << "}\n";
}
}
