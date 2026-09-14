#include "llm/mamba_state.h"
#include <algorithm>

namespace llm {

MambaState::MambaState(size_t n_layers, size_t d_inner, size_t d_state)
    : n_layers_(n_layers), d_inner_(d_inner), d_state_(d_state),
      h_(n_layers, std::vector<float>(d_inner * d_state, 0.0f)) {}

void MambaState::update(size_t layer, const std::vector<float>& h) {
    if (layer < n_layers_) h_[layer] = h;
}

const std::vector<float>& MambaState::get(size_t layer) const {
    return h_[layer];
}

void MambaState::clear() {
    for (auto& s : h_) std::fill(s.begin(), s.end(), 0.0f);
}

size_t MambaState::size() const {
    size_t count = 0;
    for (auto& s : h_)
        for (float v : s)
            if (v != 0.0f) { count++; break; }
    return count;
}

}  // namespace llm
