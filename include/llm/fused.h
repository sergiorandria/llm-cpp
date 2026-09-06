#pragma once
#include "tensor.h"
namespace llm {
// Real fused ops: single-pass, no intermediate allocation, SIMD-friendly
// fused_layernorm_residual: y = layernorm(x, gamma, beta) + residual in one loop per row (saves 1 full pass vs separate layernorm + add)
// fused_gelu: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) single-pass with OpenMP/SIMD, not just alias to x.gelu()
Tensor fused_layernorm_residual(const Tensor& x, const Tensor& residual,
                                 const Tensor* gamma, const Tensor* beta, float eps=1e-5f);
Tensor fused_gelu(const Tensor& x);
// Optional matmul+gelu fusion helper (avoids second pass over matmul output)
Tensor fused_matmul_gelu(const Tensor& a, const Tensor& b);
}
