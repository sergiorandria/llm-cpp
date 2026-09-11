#pragma once
#include "tensor.h"
#include "model.h"
namespace llm {
Tensor quantize_int8(const Tensor& x); // F51: returns real I8 Tensor (idata + scale)
Tensor dequantize_int8(const Tensor& q, float scale); // legacy float-level path (kept for compat)
Tensor dequantize_int8(const Tensor& q); // F51: uses embedded scale (requires I8)
struct QuantizedTensor {
    Tensor q; // int8 levels stored as float (legacy simulated path; Tensor I8 is the real one)
    float scale;
};
QuantizedTensor quantize_with_scale(const Tensor& x);
Tensor dequantize(const QuantizedTensor& qt);
// Model-level quantization (int8 for all linear weights)
void quantize_model(GPT& model);
float quantize_error(const Tensor& orig, const QuantizedTensor& qt);
// E50: fused int8 GEMM — out = act @ (wq*scale) computed as scale*sum(act*wq)
// without materializing a dequantized weight pass. wq holds rounded int levels.
Tensor matmul_int8(const Tensor& act, const QuantizedTensor& wqt);
}
