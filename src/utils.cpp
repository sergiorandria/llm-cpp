#include "llm/utils.h"
#include <atomic>
#include <chrono>
#include <ctime>
namespace llm {
void set_seed(unsigned seed){ srand(seed); set_global_seed(seed); }
std::string now_string(){ auto t=std::time(nullptr); char buf[64]; std::strftime(buf,sizeof(buf),"%Y-%m-%d %H:%M:%S", std::localtime(&t)); return buf; }
size_t estimate_memory(size_t n_params){ return n_params*4 + n_params*4*2; /* params + adam m+v */ }
namespace { std::atomic<uint64_t> g_seed{42}; }
void set_global_seed(uint64_t seed){ g_seed.store(seed); srand((unsigned)seed); }
uint64_t global_seed(){ return g_seed.load(); }
}
