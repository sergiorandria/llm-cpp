#include <cassert>
#include <cmath>
#include <iostream>

#include "llm/gptq.h"

int main() {
    llm::Tensor x({1, 8}, 0.0f);
    for (size_t i = 0; i < 8; ++i) x.data[i] = float(i) - 4.0f;  // -4..3
    auto q = llm::quantize_4bit(x, 4);
    assert(q.shape == x.shape);
    // Check quantized values in [-8,7]
    for (float v : q.data) assert(v >= -8.0f && v <= 7.0f);
    // Create scales per group
    llm::Tensor scale({2}, 0.0f);
    // group 4: first group max 4 => scale 4/7, second group max 3 => 3/7
    scale.data[0] = 4.0f / 7.0f + 1e-8f;
    scale.data[1] = 3.0f / 7.0f + 1e-8f;
    auto rec = llm::dequantize_4bit(q, scale, 4);
    assert(rec.shape == x.shape);
    // Dequantized should be close to original (quant error < scale)
    float err = 0;
    for (size_t i = 0; i < 8; ++i) err += std::abs(rec.data[i] - x.data[i]);
    err /= 8;
    std::cout << "gptq err " << err << "\n";
    assert(err < 1.0f);
    // Test roundtrip with actual group computation: quant then dequant via computed scale
    llm::Tensor y({2, 4}, 0.0f);
    for (size_t i = 0; i < 8; ++i) y.data[i] = (i % 2 == 0) ? 1.5f : -1.5f;
    auto q2 = llm::quantize_4bit(y, 4);
    llm::Tensor scale2({2}, 0.0f);
    scale2.data[0] = 1.5f / 7.0f + 1e-8f;
    scale2.data[1] = 1.5f / 7.0f + 1e-8f;
    auto rec2 = llm::dequantize_4bit(q2, scale2, 4);
    for (size_t i = 0; i < 8; ++i) assert(std::abs(rec2.data[i] - y.data[i]) < 0.3f);
    std::cout << "gptq test passed\n";
    return 0;
}
