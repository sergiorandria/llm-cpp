#pragma once
#include "tensor.h"
namespace llm {
Tensor quantize_int8(const Tensor& x);
Tensor dequantize_int8(const Tensor& q, float scale);
}
