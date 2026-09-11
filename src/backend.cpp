#include "llm/backend.h"
namespace llm {
Backend active_backend() {
#ifdef USE_CUDA
    return Backend::CUDA;
#else
    return Backend::CPU;
#endif
}
const char* backend_name() {
#ifdef USE_CUDA
    return "cuda";
#else
    return "cpu";
#endif
}
bool cuda_available() {
#ifdef USE_CUDA
    return true;
#else
    return false;
#endif
}
} // namespace llm
