#include "llm/quantize.h"

#include <algorithm>
namespace llm {
Tensor quantize_int8(const Tensor& x) {
    Tensor q = x;
    q.quantize_to_int8();
    return q;
}
Tensor dequantize_int8(const Tensor& q) {
    return q.dequantized();
}
Tensor dequantize_int8(const Tensor& q, float scale) {
    Tensor out(q.shape, 0);
    for (size_t i = 0; i < q.data.size(); ++i) out.data[i] = q.data[i] * scale;
    return out;
}
QuantizedTensor quantize_with_scale(const Tensor& x) {
    float maxv = 0;
    for (float v : x.data) maxv = std::max(maxv, std::abs(v));
    float scale = maxv / 127.0f + 1e-8f;
    Tensor q(x.shape, 0);
    for (size_t i = 0; i < x.data.size(); ++i) q.data[i] = std::round(x.data[i] / scale);
    return {q, scale};
}
Tensor dequantize(const QuantizedTensor& qt) {
    return dequantize_int8(qt.q, qt.scale);
}
void quantize_model(GPT& model) {
    // Quantize all linear weights in-place via dequantize-after-quantize (simulated int8)
    for (auto* p : model.parameters()) {
        auto qt = quantize_with_scale(*p);
        *p = dequantize(qt);
    }
}
float quantize_error(const Tensor& orig, const QuantizedTensor& qt) {
    Tensor rec = dequantize(qt);
    float err = 0;
    for (size_t i = 0; i < orig.data.size(); ++i) err += std::abs(orig.data[i] - rec.data[i]);
    return err / orig.data.size();
}
Tensor matmul_int8(const Tensor& act, const QuantizedTensor& wqt) {
    assert(act.shape.size() == 2 && wqt.q.shape.size() == 2);
    assert(act.shape[1] == wqt.q.shape[0]);
    Tensor out({act.shape[0], wqt.q.shape[1]}, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < act.shape[0]; ++i) {
        for (size_t k = 0; k < act.shape[1]; ++k) {
            float a = act(i, k);
            for (size_t j = 0; j < wqt.q.shape[1]; ++j) out(i, j) += a * wqt.q(k, j);
        }
    }
    for (auto& v : out.data) v *= wqt.scale;
    return out;
}
}  // namespace llm
