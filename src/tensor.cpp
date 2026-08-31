#include "llm/tensor.h"
#include <stdexcept>

namespace llm {

Tensor::Tensor(std::vector<size_t> shape_, float fill) : shape(std::move(shape_)) {
    compute_strides();
    data.assign(numel(), fill);
}

size_t Tensor::numel() const {
    if (shape.empty()) return 0;
    size_t n = 1;
    for (auto s : shape) n *= s;
    return n;
}

void Tensor::compute_strides() {
    strides.resize(shape.size());
    if (shape.empty()) return;
    strides.back() = 1;
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i+1] * shape[i+1];
    }
}

void Tensor::randn(float mean, float std) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(mean, std);
    for (auto& v : data) v = dist(rng);
}

float& Tensor::operator()(size_t i, size_t j) {
    assert(shape.size() == 2);
    return data[i * strides[0] + j * strides[1]];
}
const float& Tensor::operator()(size_t i, size_t j) const {
    assert(shape.size() == 2);
    return data[i * strides[0] + j * strides[1]];
}

Tensor Tensor::matmul(const Tensor& other) const {
    assert(shape.size() == 2 && other.shape.size() == 2);
    assert(shape[1] == other.shape[0]);
    Tensor out({shape[0], other.shape[1]}, 0.0f);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t k = 0; k < shape[1]; ++k) {
            float a = (*this)(i, k);
            for (size_t j = 0; j < other.shape[1]; ++j) {
                out(i, j) += a * other(k, j);
            }
        }
    }
    return out;
}

Tensor Tensor::transpose() const {
    assert(shape.size() == 2);
    Tensor out({shape[1], shape[0]}, 0.0f);
    for (size_t i = 0; i < shape[0]; ++i)
        for (size_t j = 0; j < shape[1]; ++j)
            out(j, i) = (*this)(i, j);
    return out;
}

Tensor Tensor::softmax(int dim) const {
    Tensor out = *this;
    if (shape.size() == 2) {
        // softmax over last dim (j)
        for (size_t i = 0; i < shape[0]; ++i) {
            float maxv = (*this)(i, 0);
            for (size_t j = 1; j < shape[1]; ++j) maxv = std::max(maxv, (*this)(i, j));
            float sum = 0;
            for (size_t j = 0; j < shape[1]; ++j) {
                out(i, j) = std::exp((*this)(i, j) - maxv);
                sum += out(i, j);
            }
            for (size_t j = 0; j < shape[1]; ++j) out(i, j) /= sum;
        }
    }
    return out;
}

Tensor Tensor::layernorm(float eps) const {
    assert(shape.size() == 2);
    Tensor out(shape, 0.0f);
    for (size_t i = 0; i < shape[0]; ++i) {
        float mean = 0;
        for (size_t j = 0; j < shape[1]; ++j) mean += (*this)(i, j);
        mean /= shape[1];
        float var = 0;
        for (size_t j = 0; j < shape[1]; ++j) {
            float d = (*this)(i, j) - mean;
            var += d * d;
        }
        var /= shape[1];
        for (size_t j = 0; j < shape[1]; ++j) {
            out(i, j) = ((*this)(i, j) - mean) / std::sqrt(var + eps);
        }
    }
    return out;
}

} // namespace llm
