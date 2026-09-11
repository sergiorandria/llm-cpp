#include "llm/tensor.h"

#include <iostream>
#include <stdexcept>

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#ifdef USE_NUMPY_CPP
#include <np/linalg.hpp>
#include <np/np.hpp>
#include <np/random.hpp>
#endif

namespace llm {

Tensor::Tensor(std::vector<size_t> shape_, float fill) : shape(std::move(shape_)) {
    compute_strides();
    data.assign(numel(), fill);
}

size_t Tensor::numel() const {
    if (shape.empty()) return 1;  // scalar
    size_t n = 1;
    for (auto s : shape) n *= s;
    return n;
}

void Tensor::compute_strides() {
    strides.resize(shape.size());
    if (shape.empty()) return;
    strides.back() = 1;
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

void Tensor::require_f32(const char* op) const {
    if (dtype == DType::I8) throw std::runtime_error(std::string(op) + " requires F32 tensor (dequantize first)");
}

void Tensor::quantize_to_int8() {
    require_f32("quantize_to_int8");
    float maxv = 0;
    for (float v : data) maxv = std::max(maxv, std::abs(v));
    i8_scale = maxv / 127.0f + 1e-8f;
    idata.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        float q = std::round(data[i] / i8_scale);
        q = std::max(-127.0f, std::min(127.0f, q));
        idata[i] = (int8_t)q;
    }
    data.clear();
    data.shrink_to_fit();
    dtype = DType::I8;
}

Tensor Tensor::dequantized() const {
    if (dtype == DType::F32) return *this;
    Tensor out(shape, 0.0f);
    out.data.resize(idata.size());
    for (size_t i = 0; i < idata.size(); ++i) out.data[i] = (float)idata[i] * i8_scale;
    return out;
}

void Tensor::randn(float mean, float std) {
    require_f32("randn");
#ifdef USE_NUMPY_CPP
    randn_np(mean, std, 42);
#else
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(mean, std);
    for (auto& v : data) v = dist(rng);
#endif
}

#ifdef USE_NUMPY_CPP
np::ndarray<float> Tensor::to_ndarray() const {
    require_f32("to_ndarray");
    std::vector<int> np_shape;
    np_shape.reserve(shape.size());
    for (auto s : shape) np_shape.push_back(static_cast<int>(s));
    // np::ndarray expects vector<int> shape
    if (np_shape.empty()) np_shape = {1};
    auto arr = np::zeros<float>(np_shape);
    // copy data (contiguous C-order)
    size_t n = std::min<size_t>(data.size(), arr.size());
    std::copy(data.begin(), data.begin() + n, arr.data().begin());
    return arr;
}

Tensor Tensor::from_ndarray(const np::ndarray<float>& arr) {
    std::vector<size_t> s;
    s.reserve(arr.shape.size());
    for (auto d : arr.shape) s.push_back(static_cast<size_t>(d));
    Tensor t(s, 0.0f);
    size_t n = std::min<size_t>(t.data.size(), arr.size());
    std::copy(arr.data().begin(), arr.data().begin() + n, t.data.begin());
    return t;
}

Tensor Tensor::matmul_np(const Tensor& other) const {
    require_f32("matmul_np");
    auto a = to_ndarray();
    auto b = other.to_ndarray();
    // np::linalg::matmul uses blocked GEMM + SIMD + threading
    auto c = np::linalg::matmul(a, b);
    return from_ndarray(c);
}

void Tensor::randn_np(float mean, float std, uint64_t seed) {
    // Use np::random::Generator (PCG64) -> standard_normal then scale
    np::random::Generator rng(seed);
    std::vector<int> np_shape;
    for (auto s : shape) np_shape.push_back(static_cast<int>(s));
    if (np_shape.empty()) np_shape = {1};
    auto arr = rng.standard_normal<float>(np_shape);
    // arr is N(0,1), scale to mean/std
    for (size_t i = 0; i < data.size() && i < arr.size(); ++i) {
        data[i] = arr.data()[i] * std + mean;
    }
}
#endif

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
    if (dtype == DType::I8 || other.dtype == DType::I8) {
        // F51: folded-scale int8 path (no materialized dequant pass).
        //ij loop with per-operand scales (F32 operand scale = 1).
        Tensor out({shape[0], other.shape[1]}, 0.0f);
        float sa = (dtype == DType::I8) ? i8_scale : 1.0f;
        float sb = (other.dtype == DType::I8) ? other.i8_scale : 1.0f;
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t k = 0; k < shape[1]; ++k) {
                float a = (dtype == DType::I8) ? (float)idata[i * shape[1] + k] : data[i * strides[0] + k * strides[1]];
                for (size_t j = 0; j < other.shape[1]; ++j) {
                    float b = (other.dtype == DType::I8) ? (float)other.idata[k * other.shape[1] + j]
                                                         : other.data[k * other.strides[0] + j * other.strides[1]];
                    out.data[i * out.shape[1] + j] += a * b;
                }
            }
        }
        for (auto& v : out.data) v *= sa * sb;
        return out;
    }
#ifdef USE_OPENBLAS
    assert(shape.size() == 2 && other.shape.size() == 2);
    assert(shape[1] == other.shape[0]);
    Tensor out({shape[0], other.shape[1]}, 0.0f);
    // cblas_sgemm RowMajor: C = alpha*A*B + beta*C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)shape[0], (int)other.shape[1], (int)shape[1],
                1.0f, data.data(), (int)shape[1],
                other.data.data(), (int)other.shape[1],
                0.0f, out.data.data(), (int)out.shape[1]);
    return out;
#elif defined(USE_NUMPY_CPP)
    // Accelerated via numpy-cpp blocked GEMM (SIMD + threading)
    assert(shape.size() == 2 && other.shape.size() == 2);
    assert(shape[1] == other.shape[0]);
    return matmul_np(other);
#else
    // OpenMP parallelized when available
    assert(shape.size() == 2 && other.shape.size() == 2);
    assert(shape[1] == other.shape[0]);
    Tensor out({shape[0], other.shape[1]}, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t k = 0; k < shape[1]; ++k) {
            float a = (*this)(i, k);
            for (size_t j = 0; j < other.shape[1]; ++j) {
                out(i, j) += a * other(k, j);
            }
        }
    }
    return out;
#endif
}

Tensor Tensor::matmul_sparse(const Tensor& other) const {
    require_f32("matmul_sparse");
    other.require_f32("matmul_sparse");
    assert(shape.size() == 2 && other.shape.size() == 2);
    assert(shape[1] == other.shape[0]);
    size_t M = shape[0], K = shape[1], N = other.shape[1];
    // CSR over B rows: for each k, list of (j, val) with val != 0
    std::vector<std::vector<std::pair<size_t, float>>> rows(K);
    for (size_t k = 0; k < K; ++k)
        for (size_t j = 0; j < N; ++j) {
            float v = other.data[k * N + j];
            if (v != 0.0f) rows[k].emplace_back(j, v);
        }
    Tensor out({M, N}, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a = data[i * K + k];
            if (a == 0.0f) continue;
            for (auto& jv : rows[k]) out.data[i * N + jv.first] += a * jv.second;
        }
    }
    return out;
}

Tensor Tensor::transpose() const {
    require_f32("transpose");
#ifdef USE_NUMPY_CPP
    // numpy-cpp: view-based transpose (shared storage, SIMD strides)
    auto a = to_ndarray();
    auto t = a.transpose();
    return from_ndarray(t);
#else
    assert(shape.size() == 2);
    Tensor out({shape[1], shape[0]}, 0.0f);
    for (size_t i = 0; i < shape[0]; ++i)
        for (size_t j = 0; j < shape[1]; ++j) out(j, i) = (*this)(i, j);
    return out;
#endif
}

Tensor Tensor::softmax(int dim) const {
    require_f32("softmax");
    Tensor out = *this;
    if (shape.size() == 2) {
// softmax over last dim (j)
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

Tensor Tensor::layernorm(const Tensor* gamma, const Tensor* beta, float eps) const {
    require_f32("layernorm");
    assert(shape.size() == 2);
    Tensor out(shape, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
            float v = ((*this)(i, j) - mean) / std::sqrt(var + eps);
            if (gamma) v *= gamma->data[j % gamma->data.size()];
            if (beta) v += beta->data[j % beta->data.size()];
            out(i, j) = v;
        }
    }
    return out;
}

Tensor Tensor::add(const Tensor& other) const {
    require_f32("add");
    assert(shape == other.shape);
    Tensor out(shape, 0.0f);
    for (size_t i = 0; i < data.size(); ++i) out.data[i] = data[i] + other.data[i];
    return out;
}
Tensor Tensor::sub(const Tensor& other) const {
    require_f32("sub");
    assert(shape == other.shape);
    Tensor out(shape, 0.0f);
    for (size_t i = 0; i < data.size(); ++i) out.data[i] = data[i] - other.data[i];
    return out;
}
Tensor Tensor::mul(const Tensor& other) const {
    require_f32("mul");
    assert(shape == other.shape);
    Tensor out(shape, 0.0f);
    for (size_t i = 0; i < data.size(); ++i) out.data[i] = data[i] * other.data[i];
    return out;
}
Tensor Tensor::scale(float s) const {
    if (dtype == DType::I8) {  // exact: fold into scale, levels untouched
        Tensor out = *this;
        out.i8_scale *= s;
        return out;
    }
    Tensor out(shape, 0.0f);
    for (size_t i = 0; i < data.size(); ++i) out.data[i] = data[i] * s;
    return out;
}
Tensor Tensor::gelu() const {
    require_f32("gelu");
    Tensor out(shape, 0.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        float x = data[i];
        out.data[i] =
            0.5f * x *
            (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265f) * (x + 0.044715f * x * x * x)));
    }
    return out;
}
Tensor Tensor::silu() const {
    require_f32("silu");
    Tensor out(shape, 0.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        float x = data[i];
        out.data[i] = x / (1.0f + std::exp(-x));
    }
    return out;
}
Tensor Tensor::dropout(float p, std::mt19937& rng) const {
    require_f32("dropout");
    if (p == 0.0f) return *this;
    Tensor out(shape, 0.0f);
    std::bernoulli_distribution dist(1.0 - p);
    float scale = 1.0f / (1.0f - p);
    for (size_t i = 0; i < data.size(); ++i) {
        out.data[i] = dist(rng) ? data[i] * scale : 0.0f;
    }
    return out;
}
float Tensor::cross_entropy(const Tensor& target) const {
    require_f32("cross_entropy");
    // naive: this = logits [N, V], target = indices [N] stored as shape [N,1] or [N]
    assert(shape.size() == 2);
    float loss = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        float maxv = (*this)(i, 0);
        for (size_t j = 1; j < shape[1]; ++j) maxv = std::max(maxv, (*this)(i, j));
        float sum = 0;
        for (size_t j = 0; j < shape[1]; ++j) sum += std::exp((*this)(i, j) - maxv);
        // target assumed one-hot? For now target.data[i] is class id if target numel==shape0
        int tgt = (int)target.data[i % target.data.size()];
        float logp = (*this)(i, tgt) - maxv - std::log(sum);
        loss -= logp;
    }
    return loss / shape[0];
}
size_t Tensor::argmax(size_t row) const {
    require_f32("argmax");
    assert(row < shape[0]);
    size_t best = 0;
    float bestv = (*this)(row, 0);
    for (size_t j = 1; j < shape[1]; ++j)
        if ((*this)(row, j) > bestv) {
            bestv = (*this)(row, j);
            best = j;
        }
    return best;
}
void Tensor::print(const std::string& name) const {
    if (!name.empty()) std::cout << name << " ";
    std::cout << "shape[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i + 1 < shape.size() ? "," : "");
    }
    std::cout << "] data[0]=" << (data.empty() ? 0 : data[0]) << "\n";
}
void Tensor::add_grad(const Tensor& g) {
    if (grad.size() != data.size()) grad.assign(data.size(), 0.0f);
    assert(g.data.size() == data.size());
    for (size_t i = 0; i < data.size(); ++i) grad[i] += g.data[i];
}
Tensor Tensor::matmul_grad_a(const Tensor& other, const Tensor& grad_out) const {
    // this = A [m,k], other = B [k,n], grad_out = dC [m,n] => dA [m,k] = dC * B^T
    assert(shape.size() == 2 && other.shape.size() == 2 && grad_out.shape.size() == 2);
    assert(shape[1] == other.shape[0]);
    assert(grad_out.shape[0] == shape[0] && grad_out.shape[1] == other.shape[1]);
    Tensor dA({shape[0], shape[1]}, 0.0f);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t k = 0; k < shape[1]; ++k) {
            float acc = 0;
            for (size_t j = 0; j < other.shape[1]; ++j) acc += grad_out(i, j) * other(k, j);
            dA(i, k) = acc;
        }
    }
    return dA;
}
Tensor Tensor::matmul_grad_b(const Tensor& other, const Tensor& grad_out) const {
    // other is A [m,k], this is B [k,n] ??? Actually called as A.matmul_grad_b(B, dC) where A is
    // this So this = A [m,k], other = B [k,n], dC [m,n] => dB [k,n] = A^T * dC But signature is
    // B.matmul_grad_b? We implement as: this = B, other = A? For symmetry, we use: A.matmul(other)
    // => dB = A^T * dC So if this is B, we need A. We'll handle both: this is B, other is A To
    // avoid confusion, we implement generic: dB = A^T * dC where A is `other` if called on B
    // Actually we will call as: B.matmul_grad_b(A, dC) is confusing. Let's implement as: this is B,
    // other is A dB(k,j) = sum_i A(i,k) * dC(i,j)
    assert(shape.size() == 2 && other.shape.size() == 2 && grad_out.shape.size() == 2);
    // this = B [k,n], other = A [m,k], grad_out = dC [m,n]
    assert(other.shape[1] == shape[0]);
    assert(grad_out.shape[0] == other.shape[0] && grad_out.shape[1] == shape[1]);
    Tensor dB({shape[0], shape[1]}, 0.0f);
    for (size_t k = 0; k < shape[0]; ++k) {
        for (size_t j = 0; j < shape[1]; ++j) {
            float acc = 0;
            for (size_t i = 0; i < other.shape[0]; ++i) acc += other(i, k) * grad_out(i, j);
            dB(k, j) = acc;
        }
    }
    return dB;
}
Tensor::LayernormGrad Tensor::layernorm_backward(const Tensor& grad_out, const Tensor* gamma,
                                                 float eps) const {
    assert(shape.size() == 2 && grad_out.shape == shape);
    size_t T = shape[0], C = shape[1];
    Tensor grad_x(shape, 0.0f);
    Tensor grad_gamma({C}, 0.0f);
    Tensor grad_beta({C}, 0.0f);
    for (size_t i = 0; i < T; ++i) {
        float mean = 0;
        for (size_t j = 0; j < C; ++j) mean += (*this)(i, j);
        mean /= C;
        float var = 0;
        for (size_t j = 0; j < C; ++j) {
            float d = (*this)(i, j) - mean;
            var += d * d;
        }
        var /= C;
        float inv_std = 1.0f / std::sqrt(var + eps);
        // grad_beta = sum grad_out
        // grad_gamma = sum grad_out * x_hat
        for (size_t j = 0; j < C; ++j) {
            float x_hat = ((*this)(i, j) - mean) * inv_std;
            grad_beta.data[j] += grad_out(i, j);
            grad_gamma.data[j] += grad_out(i, j) * x_hat;
        }
        // grad_x
        // From layernorm paper: dx = (1/sqrt(var+eps)) * ( dy*gamma - mean(dy*gamma) -
        // x_hat*mean(dy*gamma*x_hat) ) First compute gamma*dy
        float mean_dy_gamma = 0, mean_dy_gamma_xhat = 0;
        for (size_t j = 0; j < C; ++j) {
            float g = gamma ? gamma->data[j % gamma->data.size()] : 1.0f;
            float dy_g = grad_out(i, j) * g;
            mean_dy_gamma += dy_g;
            float x_hat = ((*this)(i, j) - mean) * inv_std;
            mean_dy_gamma_xhat += dy_g * x_hat;
        }
        mean_dy_gamma /= C;
        mean_dy_gamma_xhat /= C;
        for (size_t j = 0; j < C; ++j) {
            float g = gamma ? gamma->data[j % gamma->data.size()] : 1.0f;
            float dy_g = grad_out(i, j) * g;
            float x_hat = ((*this)(i, j) - mean) * inv_std;
            grad_x(i, j) = inv_std * (dy_g - mean_dy_gamma - x_hat * mean_dy_gamma_xhat);
        }
    }
    return {grad_x, grad_gamma, grad_beta};
}

Tensor Tensor::rmsnorm(const Tensor* weight, float eps) const {
    require_f32("rmsnorm");
    assert(shape.size() == 2);
    Tensor out(shape, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < shape[0]; ++i) {
        float ms = 0;
        for (size_t j = 0; j < shape[1]; ++j) ms += (*this)(i, j) * (*this)(i, j);
        ms /= (float)shape[1];
        float inv = 1.0f / std::sqrt(ms + eps);
        for (size_t j = 0; j < shape[1]; ++j) {
            float v = (*this)(i, j) * inv;
            if (weight) v *= weight->data[j % weight->data.size()];
            out(i, j) = v;
        }
    }
    return out;
}

Tensor::RmsnormGrad Tensor::rmsnorm_backward(const Tensor& grad_out, const Tensor* weight,
                                             float eps) const {
    assert(shape.size() == 2 && grad_out.shape == shape);
    size_t T = shape[0], C = shape[1];
    Tensor grad_x(shape, 0.0f);
    Tensor grad_w({C}, 0.0f);
    for (size_t i = 0; i < T; ++i) {
        float ms = 0;
        for (size_t j = 0; j < C; ++j) ms += (*this)(i, j) * (*this)(i, j);
        ms /= (float)C;
        float inv = 1.0f / std::sqrt(ms + eps);
        float inv3 = inv * inv * inv / (float)C;
        // grad_w += grad_out * x_hat (x_hat = x*inv)
        for (size_t j = 0; j < C; ++j) {
            float x_hat = (*this)(i, j) * inv;
            grad_w.data[j] += grad_out(i, j) * x_hat;
        }
        // grad_x = inv*(dy*w) - x * (sum(dy*w*x) * inv^3 / C)
        float dot = 0;
        for (size_t j = 0; j < C; ++j) {
            float w = weight ? weight->data[j % weight->data.size()] : 1.0f;
            dot += grad_out(i, j) * w * (*this)(i, j);
        }
        float coef = dot * inv3;
        for (size_t j = 0; j < C; ++j) {
            float w = weight ? weight->data[j % weight->data.size()] : 1.0f;
            grad_x(i, j) = grad_out(i, j) * w * inv - (*this)(i, j) * coef;
        }
    }
    return {grad_x, grad_w};
}

}  // namespace llm
