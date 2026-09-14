#include "llm/backend.h"
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif
namespace llm {
bool cuda_available() {
#ifdef USE_CUDA
    // Runtime probe: a USE_CUDA binary still runs (on CPU) where no GPU exists.
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
    return false;
#endif
}
Backend active_backend() {
    return cuda_available() ? Backend::CUDA : Backend::CPU;
}
const char* backend_name() {
    return cuda_available() ? "cuda" : "cpu";
}
}  // namespace llm
