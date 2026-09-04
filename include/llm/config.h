#pragma once
#include "model.h"
#include <string>
namespace llm {
Config load_config(const std::string& path);
void save_config(const Config& cfg, const std::string& path);
bool validate_config(const Config& cfg);
}
