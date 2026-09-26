#pragma once
#include <cstdint>
#include <random>
#include <string>
namespace llm {
void set_seed(unsigned seed);
std::string now_string();
size_t estimate_memory(size_t n_params);
// A06: global deterministic seed (sampling + init). Default 42.
void set_global_seed(uint64_t seed);
uint64_t global_seed();
// I81: thread pool setup from LLM_THREADS / OMP_NUM_THREADS (returns threads used)
// Also caps OpenMP nesting at 1 by default (I89 oversubscription fix) unless
// OMP_MAX_ACTIVE_LEVELS/OMP_NESTED is set.
int init_threading();
// Test hook for the nesting-cap env logic (no OpenMP needed).
int nesting_capped_default();
// I82: compile-time SIMD capability string (e.g. "AVX2+OpenMP", "NEON", "scalar")
std::string simd_caps();
}  // namespace llm
