#include "llm/utils.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace llm {
void set_seed(unsigned seed) {
    srand(seed);
    set_global_seed(seed);
}
std::string now_string() {
    auto t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}
size_t estimate_memory(size_t n_params) {
    return n_params * 4 + n_params * 4 * 2; /* params + adam m+v */
}
namespace {
std::atomic<uint64_t> g_seed{42};
}
void set_global_seed(uint64_t seed) {
    g_seed.store(seed);
    srand((unsigned)seed);
}
uint64_t global_seed() {
    return g_seed.load();
}
int init_threading() {
    int want = 0;
    if (const char* e = std::getenv("LLM_THREADS")) want = std::atoi(e);
    if (want <= 0) {
        if (const char* e = std::getenv("OMP_NUM_THREADS")) want = std::atoi(e);
    }
#ifdef _OPENMP
    if (want > 0) omp_set_num_threads(want);
    // I89 pathology: our parallel-for regions calling into threaded kernels
    // (numpy/OpenBLAS micro-GEMMs) oversubscribe — measured 87x slower with
    // default threads on a tiny model (12 vs 1078 tok/s). Cap nesting at 1
    // (inner regions run serially; outer parallelism kept) unless the user
    // explicitly configured nesting. Must run outside any parallel region.
    if (!std::getenv("OMP_MAX_ACTIVE_LEVELS") && !std::getenv("OMP_NESTED"))
        omp_set_max_active_levels(1);
    return omp_get_max_threads();
#else
    (void)want;
    return 1;
#endif
}

// Test hook: reports whether nesting would be capped (pure env logic,
// usable without OpenMP). Returns 1 if a cap applies, 0 if user overrode.
int nesting_capped_default() {
    if (std::getenv("OMP_MAX_ACTIVE_LEVELS") || std::getenv("OMP_NESTED")) return 0;
    return 1;
}
std::string simd_caps() {
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
}  // namespace llm
