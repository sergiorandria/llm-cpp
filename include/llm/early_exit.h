#pragma once
#include "tensor.h"
namespace llm {
// Early exit: if max softmax >0.9, skip remaining TransformerBlocks — 1.4× avg speedup
bool should_early_exit(const Tensor& logits, float thresh=0.9f);
}
