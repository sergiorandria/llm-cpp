#include "llm/optimizer.h"
#include <cmath>
namespace llm {
void SGD::step(std::vector<Tensor*>& params, const std::vector<Tensor>& grads){
    for(size_t i=0;i<params.size() && i<grads.size(); ++i){
        for(size_t j=0;j<params[i]->data.size(); ++j) params[i]->data[j] -= lr_ * grads[i].data[j];
    }
}
Adam::Adam(float lr, float b1, float b2, float eps): lr_(lr), b1_(b1), b2_(b2), eps_(eps){}
void Adam::step(std::vector<Tensor*>& params, const std::vector<Tensor>& grads){
    ++t_;
    if(m_.empty()){ m_=grads; v_=grads; for(auto &t: m_) std::fill(t.data.begin(), t.data.end(), 0); for(auto &t: v_) std::fill(t.data.begin(), t.data.end(), 0); }
    for(size_t i=0;i<params.size() && i<grads.size(); ++i){
        for(size_t j=0;j<params[i]->data.size(); ++j){
            m_[i].data[j]=b1_*m_[i].data[j]+(1-b1_)*grads[i].data[j];
            v_[i].data[j]=b2_*v_[i].data[j]+(1-b2_)*grads[i].data[j]*grads[i].data[j];
            float mhat=m_[i].data[j]/(1-std::pow(b1_,t_));
            float vhat=v_[i].data[j]/(1-std::pow(b2_,t_));
            params[i]->data[j] -= lr_ * mhat / (std::sqrt(vhat)+eps_);
        }
    }
}
AdamW::AdamW(float lr, float b1, float b2, float eps, float wd): Adam(lr,b1,b2,eps), wd_(wd){}
void AdamW::step(std::vector<Tensor*>& params, const std::vector<Tensor>& grads){
    for(auto p: params) for(auto &v: p->data) v -= 0.01f * wd_ * v; // naive weight decay before Adam
    Adam::step(params, grads);
}
}
