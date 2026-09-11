#pragma once
// Tracy profiling hooks — no-op if TRACY_ENABLE not defined, otherwise zone scopes
#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#define LLM_ZONE ZoneScoped
#define LLM_ZONE_N(n) ZoneScopedN(n)
#else
#define LLM_ZONE \
    do {         \
    } while (0)
#define LLM_ZONE_N(n) \
    do {              \
    } while (0)
#endif
// SoA layout note: Tensor currently AoS (vector<float> row-major), SoA would be
// struct { vector<float> data; } with column-major for matmul — future: transpose B for better
// cache

// H73: built-in scoped timers (always on, ~ns overhead). PROFILE("attn") records
// wall ms into a global registry; profiling_report() prints per-op totals.
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
namespace llm {
struct ProfEntry {
    uint64_t calls = 0;
    double ms_total = 0.0;
};
std::unordered_map<std::string, ProfEntry>& prof_registry();
std::mutex& prof_mutex();
inline void prof_add(const std::string& name, double ms) {
    std::lock_guard<std::mutex> lk(prof_mutex());
    auto& e = prof_registry()[name];
    e.calls++;
    e.ms_total += ms;
}
class ScopedTimer {
   public:
    explicit ScopedTimer(const char* n);
    ~ScopedTimer();

   private:
    const char* name_;
    uint64_t t0_;
};
uint64_t prof_now_ns();
std::string profiling_report();
}  // namespace llm
#define PROFILE(name) llm::ScopedTimer _llm_prof_timer_##__LINE__(name)
