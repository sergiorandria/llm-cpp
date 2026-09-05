#pragma once
#include "tensor.h"
namespace llm {
// 4-bit GPTQ quantization — group_size 128, 4× memory, <0.3 PPL loss at 7B
Tensor quantize_4bit(const Tensor& x, size_t group=128);
Tensor dequantize_4bit(const Tensor& q, const Tensor& scale, size_t group=128);
}
