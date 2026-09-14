#pragma once
#include <vector>

namespace llm {

class MambaState {
public:
    MambaState(size_t n_layers, size_t d_inner, size_t d_state);

    void update(size_t layer, const std::vector<float>& h);
    const std::vector<float>& get(size_t layer) const;
    void clear();
    size_t size() const;
    size_t n_layers() const { return n_layers_; }
    size_t d_inner() const { return d_inner_; }
    size_t d_state() const { return d_state_; }

private:
    size_t n_layers_, d_inner_, d_state_;
    std::vector<std::vector<float>> h_;
};

}  // namespace llm
