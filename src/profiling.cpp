#include "llm/profiling.h"
#include <chrono>
#include <sstream>
namespace llm {
std::unordered_map<std::string, ProfEntry>& prof_registry() {
    static std::unordered_map<std::string, ProfEntry> r;
    return r;
}
std::mutex& prof_mutex() {
    static std::mutex m;
    return m;
}
uint64_t prof_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
ScopedTimer::ScopedTimer(const char* n) : name_(n), t0_(prof_now_ns()) {}
ScopedTimer::~ScopedTimer() {
    prof_add(name_, (prof_now_ns() - t0_) / 1e6);
}
std::string profiling_report() {
    std::lock_guard<std::mutex> lk(prof_mutex());
    std::ostringstream o;
    for (auto& kv : prof_registry())
        o << kv.first << ": calls=" << kv.second.calls << " ms=" << kv.second.ms_total << "\n";
    return o.str();
}
} // namespace llm
