#pragma once
#include <vector>
#include <cstddef>
#include <cassert>
#include <algorithm>
#include <random>
#include <cmath>

namespace llm {

class Tensor {
public:
    std::vector<float> data;
    std::vector<float> grad; // for autograd stub
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    Tensor() = default;
    explicit Tensor(std::vector<size_t> shape_, float fill = 0.0f);

    size_t numel() const;
    size_t ndim() const { return shape.size(); }

    void fill(float v) { std::fill(data.begin(), data.end(), v); }
    void zero_grad(){ if(grad.size()!=data.size()) grad.assign(data.size(),0.0f); else std::fill(grad.begin(), grad.end(), 0.0f); }
    void randn(float mean = 0.0f, float std = 0.02f);

    float& operator()(size_t i, size_t j);
    const float& operator()(size_t i, size_t j) const;

    // Ops (CPU naive)
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    Tensor softmax(int dim = -1) const;
    Tensor layernorm(float eps = 1e-5f) const;
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const; // elementwise
    Tensor scale(float s) const;
    Tensor gelu() const;
    Tensor silu() const;
    Tensor dropout(float p, std::mt19937& rng) const;
    float cross_entropy(const Tensor& target) const; // assume logits
    size_t argmax(size_t row) const;

    static Tensor zeros(std::vector<size_t> shape) { return Tensor(shape, 0.0f); }
    static Tensor ones(std::vector<size_t> shape) { return Tensor(shape, 1.0f); }
    void print(const std::string& name="") const;
    std::vector<size_t> get_shape() const { return shape; }

private:
    void compute_strides();
    size_t offset(const std::vector<size_t>& indices) const;
};

} // namespace llm
