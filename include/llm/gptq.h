#pragma once
#include <string>
#include <utility>
#include <vector>

#include "tensor.h"
namespace llm {
class GPT;  // fwd
// 4-bit GPTQ quantization — group_size 128, 4× memory, <0.3 PPL loss at 7B
// Original overload (for compatibility) — computes scale internally and discards (fragile)
// New overload outputs per-group scale tensor needed for dequantize.
Tensor quantize_4bit(const Tensor& x, size_t group = 128);
Tensor quantize_4bit(const Tensor& x, Tensor& scale_out, size_t group = 128);
Tensor dequantize_4bit(const Tensor& q, const Tensor& scale, size_t group = 128);
// F52: whole-model 4-bit with scale persistence (sidecar <prefix>.gptq.bin)
struct GPTQEntry {
    std::string name;
    Tensor q;
    Tensor scale;
    size_t group = 128;
};
std::vector<GPTQEntry> quantize_model_4bit(GPT& model, size_t group = 128);
void save_gptq(const std::string& path, const std::vector<GPTQEntry>& entries);
std::vector<GPTQEntry> load_gptq(const std::string& path);
// F53: AWQ — activation-aware per-channel rescale. Returns (scaled_weight, inv_scales)
// so that scaled*inv == orig in fp; quantizing scaled protects salient channels.
std::pair<Tensor, Tensor> awq_rescale_for_quant(const Tensor& weight, const Tensor& act_mag,
                                                float alpha = 0.5f);
Tensor channel_act_mag(const Tensor& activations);  // mean|.| per input channel [C]
}  // namespace llm
