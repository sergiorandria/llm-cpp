#pragma once
#include "tensor.h"
namespace llm {
// Activation checkpointing: recompute forward on backward to save O(n_layers) memory
// Trade 1 extra forward for 1/n_layers memory — useful for 12+ layers on edge
struct CheckpointConfig { bool enabled=false; size_t segment=2; };
Tensor checkpointed_forward(const Tensor& x, const CheckpointConfig& cfg);
}
