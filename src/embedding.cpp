#include "llm/embedding.h"
namespace llm {
Embedding::Embedding(size_t vocab, size_t dim): vocab_(vocab), dim_(dim), weight_({vocab, dim}){ weight_.randn(0,0.02f); }
Tensor Embedding::forward(const std::vector<int>& ids) const {
    Tensor out({ids.size(), dim_},0);
    for(size_t i=0;i<ids.size();++i) for(size_t j=0;j<dim_;++j) out(i,j)=weight_(ids[i]%vocab_, j);
    return out;
}
}
