#include "llm/sampling.h"
#include <cassert>
#include <iostream>
int main(){
    std::vector<float> logits={0,1,2,3};
    int g=llm::sample_greedy(logits); assert(g==3);
    int k=llm::sample_top_k(logits,2,1.0f); assert(k==2||k==3);
    int p=llm::sample_top_p(logits,0.9f,1.0f);
    (void)p;
    std::cout<<"sampling tests passed\n";
    return 0;
}
