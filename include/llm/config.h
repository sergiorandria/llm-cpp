#pragma once
#include "model.h"
#include <string>
namespace llm {
Config load_config(const std::string& path);
void save_config(const Config& cfg, const std::string& path);
bool validate_config(const Config& cfg);
// D33: HF config.json compat (n_layer/n_head/n_embd/n_positions/n_vocab -> Config)
Config load_hf_config(const std::string& path);
}
