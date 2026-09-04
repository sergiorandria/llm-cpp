#pragma once
#include <string>
#include <unordered_map>
namespace llm {
struct Args {
    std::unordered_map<std::string,std::string> kv;
    std::string get(const std::string& k, const std::string& def="") const;
    bool has(const std::string& k) const;
};
Args parse_args(int argc, char* argv[]);
}
