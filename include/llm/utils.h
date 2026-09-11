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
}
