#include "llm/sampling.h"
#include <algorithm>
#include <random>
#include <cmath>
namespace llm {
int sample_greedy(const std::vector<float>& logits){ return std::max_element(logits.begin(), logits.end()) - logits.begin(); }
int sample_temperature(const std::vector<float>& logits, float temp){
    std::vector<float> l=logits;
    for(auto &v: l) v/=temp;
    float maxv=*std::max_element(l.begin(), l.end());
    float sum=0; for(auto &v:l){ v=std::exp(v-maxv); sum+=v; } for(auto &v:l) v/=sum;
    std::mt19937 rng(42); std::discrete_distribution<int> dist(l.begin(), l.end()); return dist(rng);
}
int sample_top_k(const std::vector<float>& logits, int k, float temp){
    std::vector<int> idx(logits.size()); for(size_t i=0;i<idx.size();++i) idx[i]=i;
    std::vector<float> tmp=logits; for(auto &v:tmp) v/=temp;
    std::partial_sort(idx.begin(), idx.begin()+k, idx.end(), [&](int a,int b){return tmp[a]>tmp[b];});
    std::vector<float> topk(k); for(int i=0;i<k;++i) topk[i]=tmp[idx[i]];
    float maxv=*std::max_element(topk.begin(), topk.end());
    float sum=0; for(auto &v:topk){v=std::exp(v-maxv); sum+=v;} for(auto &v:topk) v/=sum;
    std::mt19937 rng(42); std::discrete_distribution<int> dist(topk.begin(), topk.end()); return idx[dist(rng)];
}
int sample_top_p(const std::vector<float>& logits, float p, float temp){
    std::vector<float> l=logits; for(auto &v:l) v/=temp;
    float maxv=*std::max_element(l.begin(), l.end());
    float sum=0; for(auto &v:l){v=std::exp(v-maxv); sum+=v;} for(auto &v:l) v/=sum;
    std::vector<int> idx(l.size()); for(size_t i=0;i<idx.size();++i) idx[i]=i;
    std::sort(idx.begin(), idx.end(), [&](int a,int b){return l[a]>l[b];});
    float cum=0; int last=0;
    for(size_t i=0;i<idx.size();++i){ cum+=l[idx[i]]; last=i; if(cum>=p) break; }
    std::vector<float> filt; filt.reserve(last+1); for(int i=0;i<=last;++i) filt.push_back(l[idx[i]]);
    float s=0; for(auto v:filt) s+=v; for(auto &v:filt) v/=s;
    std::mt19937 rng(42); std::discrete_distribution<int> dist(filt.begin(), filt.end()); return idx[dist(rng)];
}
std::vector<float> apply_repetition_penalty(const std::vector<float>& logits, const std::vector<int>& gen, float penalty){
    std::vector<float> out=logits;
    for(int id: gen) if(id>=0 && id<(int)out.size()){ if(out[id]>0) out[id]/=penalty; else out[id]*=penalty; }
    return out;
}
}
