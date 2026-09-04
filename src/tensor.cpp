#include "llm/tensor.h"
#include <stdexcept>
#include <iostream>

#ifdef USE_NUMPY_CPP
#include <np/np.hpp>
#include <np/linalg.hpp>
#include <np/random.hpp>
#endif

namespace llm {

Tensor::Tensor(std::vector<size_t> shape_, float fill) : shape(std::move(shape_)) {
    compute_strides();
    data.assign(numel(), fill);
}

size_t Tensor::numel() const {
    if (shape.empty()) return 1; // scalar
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
#ifdef USE_NUMPY_CPP
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
            if(gamma) v *= gamma->data[j % gamma->data.size()];
            if(beta) v += beta->data[j % beta->data.size()];
            out(i, j) = v;
        }
    }
    return out;
}

Tensor Tensor::add(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor out(shape, 0.0f);
    for (size_t i=0;i<data.size();++i) out.data[i]=data[i]+other.data[i];
    return out;
}
Tensor Tensor::sub(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor out(shape, 0.0f);
    for (size_t i=0;i<data.size();++i) out.data[i]=data[i]-other.data[i];
    return out;
}
Tensor Tensor::mul(const Tensor& other) const {
    assert(shape == other.shape);
    Tensor out(shape, 0.0f);
    for (size_t i=0;i<data.size();++i) out.data[i]=data[i]*other.data[i];
    return out;
}
Tensor Tensor::scale(float s) const {
    Tensor out(shape, 0.0f);
    for (size_t i=0;i<data.size();++i) out.data[i]=data[i]*s;
    return out;
}
Tensor Tensor::gelu() const {
    Tensor out(shape, 0.0f);
    for (size_t i=0;i<data.size();++i){
        float x=data[i];
        out.data[i]=0.5f*x*(1.0f+std::tanh(std::sqrt(2.0f/3.14159265f)*(x+0.044715f*x*x*x)));
    }
    return out;
}
Tensor Tensor::silu() const {
    Tensor out(shape, 0.0f);
    for (size_t i=0;i<data.size();++i){
        float x=data[i];
        out.data[i]=x/(1.0f+std::exp(-x));
    }
    return out;
}
Tensor Tensor::dropout(float p, std::mt19937& rng) const {
    if(p==0.0f) return *this;
    Tensor out(shape, 0.0f);
    std::bernoulli_distribution dist(1.0-p);
    float scale = 1.0f/(1.0f-p);
    for(size_t i=0;i<data.size();++i){
        out.data[i]= dist(rng) ? data[i]*scale : 0.0f;
    }
    return out;
}
float Tensor::cross_entropy(const Tensor& target) const {
    // naive: this = logits [N, V], target = indices [N] stored as shape [N,1] or [N]
    assert(shape.size()==2);
    float loss=0;
    for(size_t i=0;i<shape[0];++i){
        float maxv = (*this)(i,0);
        for(size_t j=1;j<shape[1];++j) maxv = std::max(maxv, (*this)(i,j));
        float sum=0;
        for(size_t j=0;j<shape[1];++j) sum+= std::exp((*this)(i,j)-maxv);
        // target assumed one-hot? For now target.data[i] is class id if target numel==shape0
        int tgt = (int)target.data[i % target.data.size()];
        float logp = (*this)(i,tgt)-maxv - std::log(sum);
        loss -= logp;
    }
    return loss / shape[0];
}
size_t Tensor::argmax(size_t row) const {
    assert(row < shape[0]);
    size_t best=0; float bestv=(*this)(row,0);
    for(size_t j=1;j<shape[1];++j) if((*this)(row,j) > bestv){ bestv=(*this)(row,j); best=j; }
    return best;
}
void Tensor::print(const std::string& name) const {
    if(!name.empty()) std::cout<<name<<" ";
    std::cout<<"shape[";
    for(size_t i=0;i<shape.size();++i){ std::cout<<shape[i]<<(i+1<shape.size()?",":""); }
    std::cout<<"] data[0]="<<(data.empty()?0:data[0])<<"\n";
}

} // namespace llm

