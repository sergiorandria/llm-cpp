#include "llm/logging.h"
namespace llm {
void log(LogLevel lvl, const std::string& msg){
    const char* p="INFO";
    if(lvl==LogLevel::DEBUG) p="DEBUG";
    else if(lvl==LogLevel::WARN) p="WARN";
    else if(lvl==LogLevel::ERROR) p="ERROR";
    std::cout<<"["<<p<<"] "<<msg<<"\n";
}
}
