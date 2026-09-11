#pragma once
#include "tensor.h"
#include <string>
#include <utility>
#include <vector>
namespace llm {
class GPT; // fwd (model.h) to avoid include cycle
// Activation checkpointing: recompute forward on backward to save O(n_layers) memory
// Trade 1 extra forward for 1/n_layers memory — useful for 12+ layers on edge
struct CheckpointConfig { bool enabled=false; size_t segment=2; };
Tensor checkpointed_forward(const Tensor& x, const CheckpointConfig& cfg);
// D32: SafeTensors (F32 only) — JSON header + LE data. Order-independent by name.
void save_safetensors(const std::string& path,
                      const std::vector<std::pair<std::string, Tensor*>>& named);
void load_safetensors(const std::string& path,
                      const std::vector<std::pair<std::string, Tensor*>>& named);
// Named view of a GPT in parameters() order: wte,wpe,ln_f_*,[lm_head],blk{i}.*.
std::vector<std::pair<std::string, Tensor*>> gpt_named_params(GPT& m);
// D38: best-checkpoint keeper — keeps top-N lowest-ppl checkpoints + best link
class BestKeeper {
public:
    BestKeeper(const std::string& dir, size_t keep_n = 3);
    // returns true if this checkpoint entered top-N (caller saves model file itself)
    bool consider(int step, float ppl);
    std::string best_path() const { return dir_ + "/best_ppl.bin"; }
    float best_ppl() const { return best_; }
private:
    std::string dir_;
    size_t keep_n_;
    float best_ = 1e30f;
    std::vector<std::pair<float, std::string>> kept_; // (ppl, path) sorted asc
};
}
