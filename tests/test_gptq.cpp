#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/gptq.h"

int main() {
    llm::Tensor x({1, 8}, 0.0f);
    for (size_t i = 0; i < 8; ++i) x.data[i] = float(i) - 4.0f;  // -4..3
    // New overload: also outputs per-group scale
    llm::Tensor scale;
    auto q = llm::quantize_4bit(x, scale, 4);
    assert(q.shape == x.shape);
    assert(scale.shape.size()==1 && scale.data.size()==2);
    // Check quantized values in [-8,7]
    for (float v : q.data) assert(v >= -8.0f && v <= 7.0f);
    // Scales should be 4/7 and 3/7
    assert(std::abs(scale.data[0] - (4.0f/7.0f+1e-8f)) < 1e-6);
    assert(std::abs(scale.data[1] - (3.0f/7.0f+1e-8f)) < 1e-6);
    // Dequantize with *returned* scale (not re-derived)
    auto rec = llm::dequantize_4bit(q, scale, 4);
    assert(rec.shape == x.shape);
    float err = 0;
    for (size_t i = 0; i < 8; ++i) err += std::abs(rec.data[i] - x.data[i]);
    err /= 8;
    std::cout << "gptq err " << err << " scale0 " << scale.data[0] << " scale1 " << scale.data[1] << "\n";
    assert(err < 0.5f && "reconstruction error should be <0.5 for 4-bit group 4");
    // Test roundtrip with y using new overload as well
    llm::Tensor y({2, 4}, 0.0f);
    for (size_t i = 0; i < 8; ++i) y.data[i] = (i % 2 == 0) ? 1.5f : -1.5f;
    llm::Tensor scale2;
    auto q2 = llm::quantize_4bit(y, scale2, 4);
    assert(scale2.data.size()==2);
    auto rec2 = llm::dequantize_4bit(q2, scale2, 4);
    for (size_t i = 0; i < 8; ++i) assert(std::abs(rec2.data[i] - y.data[i]) < 0.3f);
    std::cout << "gptq roundtrip with returned scale passed\n";

    // Also test original overload still works (discards scale) — ensure not broken
    auto q3 = llm::quantize_4bit(x, 4);
    llm::Tensor scale3({2}, 0.0f);
    scale3.data[0] = 4.0f/7.0f+1e-8f; scale3.data[1]=3.0f/7.0f+1e-8f;
    auto rec3 = llm::dequantize_4bit(q3, scale3, 4);
    float err3=0; for(size_t i=0;i<8;++i) err3+=std::abs(rec3.data[i]-x.data[i]); err3/=8;
    assert(err3 < 1.0f);
    std::cout << "gptq legacy overload still works err " << err3 << "\n";

    // Test group 128 default (larger tensor)
    llm::Tensor big({1, 256}, 0.0f);
    for(size_t i=0;i<256;++i) big.data[i]= (float)(i%16)-8.0f;
    llm::Tensor scale_big;
    auto qb = llm::quantize_4bit(big, scale_big, 128);
    assert(scale_big.data.size()==2);
    auto recb = llm::dequantize_4bit(qb, scale_big, 128);
    float errb=0; for(size_t i=0;i<256;++i) errb+=std::abs(recb.data[i]-big.data[i]); errb/=256;
    std::cout << "gptq big group128 err " << errb << "\n";
    assert(errb < 0.6f);

    std::cout << "gptq test passed\n";
    return 0;
}
