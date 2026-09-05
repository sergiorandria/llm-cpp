#pragma once
#include "tensor.h"
namespace llm {
// Fused layernorm + residual: y = layernorm(x, gamma, beta) + residual
// Saves one pass over data and is SIMD-friendly via numpy bridge
Tensor fused_layernorm_residual(const Tensor& x, const Tensor& residual,
                                const Tensor* gamma, const Tensor* beta, float eps=1e-5f);
Tensor fused_gelu(const Tensor& x);
}
