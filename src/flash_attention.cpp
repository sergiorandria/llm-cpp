#include "llm/flash_attention.h"
#include <cmath>
#include <algorithm>
namespace llm {
Tensor flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                       float scale, bool causal, size_t block_size) {
    size_t T = Q.shape[0];
    size_t D = Q.shape[1];
    // Tiled: process queries in blocks, compute scores block-wise to avoid full [T,T] mat
    Tensor out({T, D}, 0.0f);
    // For each query block
    for (size_t qb=0; qb<T; qb+=block_size) {
        size_t qe = std::min(qb+block_size, T);
        for (size_t i=qb; i<qe; ++i) {
            // Compute scores for row i against all keys up to i if causal
            size_t k_max = causal ? i+1 : T;
            // Find max for numerical stability (block-wise)
            float maxv = -1e9f;
            for (size_t j=0;j<k_max;++j) {
                float s=0; for(size_t d=0;d<D;++d) s += Q(i,d)*K(j,d);
                s*=scale; maxv = std::max(maxv, s);
            }
            float sum=0;
            // Compute exp and sum
            std::vector<float> exps(k_max);
            for (size_t j=0;j<k_max;++j){
                float s=0; for(size_t d=0;d<D;++d) s += Q(i,d)*K(j,d);
                s*=scale;
                float e = std::exp(s - maxv);
                exps[j]=e; sum+=e;
            }
            // Weighted sum of V
            for (size_t d=0; d<D; ++d){
                float acc=0;
                for (size_t j=0;j<k_max;++j) acc += (exps[j]/sum) * V(j,d);
                out(i,d)=acc;
            }
        }
    }
    return out;
}

Tensor flash_attention_incremental(const Tensor& Q, const Tensor& K, const Tensor& V,
                                   float scale, size_t block_size) {
    assert(Q.shape[0]==1);
    size_t K_len = K.shape[0];
    size_t D = Q.shape[1];
    assert(K.shape[1]==D && V.shape[1]==D && V.shape[0]==K_len);
    Tensor out({1, D}, 0.0f);
    // Single query tiled over K blocks
    float maxv = -1e9f;
    // First pass max (block-wise)
    for(size_t kb=0; kb<K_len; kb+=block_size){
        size_t ke = std::min(kb+block_size, K_len);
        for(size_t j=kb;j<ke;++j){
            float s=0; for(size_t d=0;d<D;++d) s+= Q(0,d)*K(j,d);
            s*=scale; maxv = std::max(maxv, s);
        }
    }
    float sum=0;
    std::vector<float> exps(K_len);
    for(size_t kb=0; kb<K_len; kb+=block_size){
        size_t ke = std::min(kb+block_size, K_len);
        for(size_t j=kb;j<ke;++j){
            float s=0; for(size_t d=0;d<D;++d) s+= Q(0,d)*K(j,d);
            s*=scale;
            float e = std::exp(s - maxv);
            exps[j]=e; sum+=e;
        }
    }
    for(size_t d=0; d<D; ++d){
        float acc=0;
        for(size_t j=0;j<K_len;++j) acc += (exps[j]/sum) * V(j,d);
        out(0,d)=acc;
    }
    return out;
}
}
