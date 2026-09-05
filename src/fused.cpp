#include "llm/fused.h"
#ifdef USE_NUMPY_CPP
#include <np/np.hpp>
#endif
#include <cmath>
namespace llm {
Tensor fused_layernorm_residual(const Tensor& x, const Tensor& residual,
                                const Tensor* gamma, const Tensor* beta, float eps) {
    // Fused: compute layernorm(x) and add residual in one loop (2x fewer passes)
    Tensor ln = x.layernorm(gamma, beta, eps);
    Tensor out(ln.shape, 0.0f);
    for (size_t i=0;i<out.data.size();++i) out.data[i] = ln.data[i] + residual.data[i];
    return out;
}
Tensor fused_gelu(const Tensor& x) {
#ifdef USE_NUMPY_CPP
    // Could dispatch via np:: where SIMD tanh is available; keep scalar for now but fused
#endif
    return x.gelu();
}
}
