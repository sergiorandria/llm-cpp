#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "llm/quantize.h"
int main() {
    llm::Tensor x({4, 4}, 0.0f);
    for (size_t i = 0; i < x.data.size(); ++i)
        x.data[i] = (i % 2 ? 0.5f : -0.5f) * (float)(i * 0.1);
    // F51: real I8 storage
    llm::Tensor q = llm::quantize_int8(x);
    assert(q.is_int8());
    assert(q.idata.size() == x.shape[0] * x.shape[1]);
    assert(q.data.empty());
    for (auto v : q.idata) assert(v >= -127 && v <= 127);
    auto rec = llm::dequantize_int8(q);
    assert(!rec.is_int8() && rec.shape == x.shape);
    float err = 0;
    for (size_t i = 0; i < 16; ++i) err += std::fabs(rec.data[i] - x.data[i]);
    err /= 16;
    std::cout << "i8 err=" << err << " scale=" << q.i8_scale << "\n";
    assert(err < 0.1f);
    // I8 matmul == fp32 within quant bound
    llm::Tensor A({2, 4}, 0.0f);
    for (size_t i = 0; i < A.data.size(); ++i) A.data[i] = (float)i * 0.25f - 0.5f;
    llm::Tensor W({4, 3}, 0.0f);
    for (size_t i = 0; i < W.data.size(); ++i) W.data[i] = (float)(i % 5) * 0.2f - 0.3f;
    llm::Tensor Wq = W;
    Wq.quantize_to_int8();
    auto y_q = A.matmul(Wq);
    auto y_fp = A.matmul(W);
    assert(!y_q.is_int8());
    float md = 0;
    for (size_t i = 0; i < y_q.data.size(); ++i)
        md = std::max(md, std::fabs(y_q.data[i] - y_fp.data[i]));
    std::cout << "i8 matmul maxd=" << md << "\n";
    assert(md < 0.15f);
    // fp32-only ops reject I8 with clear error
    bool threw = false;
    try {
        (void)Wq.layernorm(nullptr, nullptr);
    } catch (const std::runtime_error& e) {
        threw = true;
    }
    assert(threw);
    threw = false;
    try {
        (void)Wq.softmax();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    // scale() on I8 folds into scale exactly
    auto qs = Wq.scale(2.0f);
    assert(qs.is_int8() && std::fabs(qs.i8_scale - 2 * Wq.i8_scale) < 1e-7);
    // I88: I8xI8 integer MACs == fp32 within quant bound
    llm::Tensor Aq = A;
    Aq.quantize_to_int8();
    auto y_qq = Aq.matmul(Wq);
    float md2 = 0;
    for (size_t i = 0; i < y_qq.data.size(); ++i)
        md2 = std::max(md2, std::fabs(y_qq.data[i] - y_fp.data[i]));
    std::cout << "i8xi8 maxd=" << md2 << "\n";
    assert(md2 < 0.3f);
    std::cout << "int8 dtype test passed\n";
    return 0;
}
