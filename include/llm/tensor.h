#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef USE_NUMPY_CPP
#include <np/linalg.hpp>
#include <np/np.hpp>
#include <np/random.hpp>
#include <np/statistics.hpp>
#endif

namespace llm {

// F51: storage dtype. I8 holds rounded int levels in idata + per-tensor scale;
// data is empty for I8 (use dequantized() for an F32 copy). Only matmul and the
// quantize/dequantize helpers accept I8; other ops throw with a clear message.
enum class DType { F32, I8 };

class Tensor {
   public:
    std::vector<float> data;
    mutable std::vector<float> grad;  // for autograd (mutable so const backward can accumulate)
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    DType dtype = DType::F32;
    std::vector<int8_t> idata;  // valid iff dtype == I8
    float i8_scale = 1.0f;      // valid iff dtype == I8

    Tensor() = default;
    explicit Tensor(std::vector<size_t> shape_, float fill = 0.0f);

    size_t numel() const;
    size_t ndim() const {
        return shape.size();
    }

    void fill(float v) {
        std::fill(data.begin(), data.end(), v);
    }
    void zero_grad() {
        if (grad.size() != data.size())
            grad.assign(data.size(), 0.0f);
        else
            std::fill(grad.begin(), grad.end(), 0.0f);
    }
    void randn(float mean = 0.0f, float std = 0.02f);

    float& operator()(size_t i, size_t j);
    const float& operator()(size_t i, size_t j) const;

    // Ops (CPU naive)
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    Tensor softmax(int dim = -1) const;
    Tensor layernorm(const Tensor* gamma = nullptr, const Tensor* beta = nullptr,
                     float eps = 1e-5f) const;
    Tensor rmsnorm(const Tensor* weight = nullptr, float eps = 1e-5f) const;
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;  // elementwise
    Tensor scale(float s) const;
    Tensor gelu() const;
    Tensor silu() const;
    Tensor dropout(float p, std::mt19937& rng) const;
    float cross_entropy(const Tensor& target) const;  // assume logits
    size_t argmax(size_t row) const;

    static Tensor zeros(std::vector<size_t> shape) {
        return Tensor(shape, 0.0f);
    }
    static Tensor ones(std::vector<size_t> shape) {
        return Tensor(shape, 1.0f);
    }
    // ── F51 int8 helpers ──
    bool is_int8() const { return dtype == DType::I8; }
    void require_f32(const char* op) const;  // throws std::runtime_error on I8
    void quantize_to_int8();                  // F32 -> I8 in place (per-tensor scale)
    Tensor dequantized() const;               // I8 -> F32 copy (throws if already F32? no: returns *this)
    void print(const std::string& name = "") const;
    std::vector<size_t> get_shape() const {
        return shape;
    }
    // Autograd helpers (manual backward, mirrors forward)
    void add_grad(const Tensor& g);                                           // grad += g
    Tensor matmul_grad_a(const Tensor& other, const Tensor& grad_out) const;  // dA
    Tensor matmul_grad_b(const Tensor& other, const Tensor& grad_out) const;  // dB
    struct LayernormGrad;
    LayernormGrad layernorm_backward(const Tensor& grad_out, const Tensor* gamma,
                                     float eps = 1e-5f) const;
    struct RmsnormGrad;
    RmsnormGrad rmsnorm_backward(const Tensor& grad_out, const Tensor* weight,
                                 float eps = 1e-5f) const;

#ifdef USE_NUMPY_CPP
    // ── numpy-cpp interop ───────────────────────────────────────────────
    np::ndarray<float> to_ndarray() const;
    static Tensor from_ndarray(const np::ndarray<float>& arr);
    // accelerated ops via numpy-cpp (SIMD, blocked GEMM, threading)
    Tensor matmul_np(const Tensor& other) const;
    void randn_np(float mean = 0.0f, float std = 0.02f, uint64_t seed = 42);
#endif

   private:
    void compute_strides();
    size_t offset(const std::vector<size_t>& indices) const;
};

struct Tensor::LayernormGrad {
    Tensor grad_x;
    Tensor grad_gamma;
    Tensor grad_beta;
};

struct Tensor::RmsnormGrad {
    Tensor grad_x;
    Tensor grad_w;
};

}  // namespace llm
