#include "llm/layernorm.h"
namespace llm {
LayerNorm::LayerNorm(size_t dim, float eps): gamma_({dim},1.0f), beta_({dim},0.0f), eps_(eps){}
Tensor LayerNorm::forward(const Tensor& x) const { return x.layernorm(&gamma_, &beta_, eps_); }
RMSNorm::RMSNorm(size_t dim, float eps): weight_({dim},1.0f), eps_(eps){}
Tensor RMSNorm::forward(const Tensor& x) const { return x.rmsnorm(&weight_, eps_); }
}
