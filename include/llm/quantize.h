#pragma once
#include "tensor.h"
#include "model.h"
namespace llm {
Tensor quantize_int8(const Tensor& x);
Tensor dequantize_int8(const Tensor& q, float scale);
struct QuantizedTensor {
    Tensor q; // int8 stored as float for stub (real would be int8)
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
