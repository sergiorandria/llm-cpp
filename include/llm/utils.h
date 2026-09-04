#pragma once
#include <random>
#include <string>
namespace llm {
void set_seed(unsigned seed);
std::string now_string();
size_t estimate_memory(size_t n_params);
}
