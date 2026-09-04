#include "llm/pos_embedding.h"
#include <cmath>
namespace llm {
Tensor sinusoidal_pos_embedding(size_t seq_len, size_t dim){
    Tensor pe({seq_len, dim},0);
    for(size_t pos=0; pos<seq_len; ++pos)
        for(size_t i=0;i<dim;++i){
            float angle = pos / std::pow(10000.0f, (2*(i/2))/ (float)dim);
            pe(pos,i) = (i%2==0) ? std::sin(angle) : std::cos(angle);
        }
    return pe;
}
PosEmbedding::PosEmbedding(size_t max_len, size_t dim, bool learned): weight_({max_len, dim}), learned_(learned){
    if(learned) weight_.randn(0,0.02f);
    else weight_=sinusoidal_pos_embedding(max_len, dim);
}
Tensor PosEmbedding::forward(size_t seq_len) const {
    Tensor out({seq_len, weight_.shape[1]},0);
    for(size_t i=0;i<seq_len;++i) for(size_t j=0;j<weight_.shape[1];++j) out(i,j)=weight_(i,j);
    return out;
}
}
