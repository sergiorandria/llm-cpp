#include "llm/utils.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace llm {
void set_seed(unsigned seed){ srand(seed); set_global_seed(seed); }
std::string now_string(){ auto t=std::time(nullptr); char buf[64]; std::strftime(buf,sizeof(buf),"%Y-%m-%d %H:%M:%S", std::localtime(&t)); return buf; }
size_t estimate_memory(size_t n_params){ return n_params*4 + n_params*4*2; /* params + adam m+v */ }
namespace { std::atomic<uint64_t> g_seed{42}; }
void set_global_seed(uint64_t seed){ g_seed.store(seed); srand((unsigned)seed); }
uint64_t global_seed(){ return g_seed.load(); }
int init_threading(){
    int want = 0;
    if (const char* e = std::getenv("LLM_THREADS")) want = std::atoi(e);
    if (want <= 0) {
        if (const char* e = std::getenv("OMP_NUM_THREADS")) want = std::atoi(e);
    }
#ifdef _OPENMP
    if (want > 0) omp_set_num_threads(want);
    return omp_get_max_threads();
#else
    (void)want;
    return 1;
#endif
}
std::string simd_caps(){
    std::string s;
#if defined(__AVX512F__)
    s += "AVX512";
#elif defined(__AVX2__)
    s += "AVX2";
#elif defined(__SSE4_2__)
    s += "SSE4.2";
#elif defined(__ARM_NEON)
    s += "NEON";
#else
    s += "scalar";
#endif
#ifdef _OPENMP
    s += "+OpenMP";
#endif
#ifdef USE_NUMPY_CPP
    s += "+numpy-cpp";
#endif
    return s;
}
}
