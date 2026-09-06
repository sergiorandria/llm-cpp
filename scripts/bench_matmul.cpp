#include "llm/tensor.h"
#ifdef USE_NUMPY_CPP
#include <np/linalg.hpp>
#endif
#include <chrono>
#include <iostream>
int main(){
    llm::Tensor A({128,128}, 0.5f); A.randn(0,1);
    llm::Tensor B({128,128}, 0.5f); B.randn(0,1);
    auto t0=std::chrono::high_resolution_clock::now();
    auto C=A.matmul(B);
    auto t1=std::chrono::high_resolution_clock::now();
    std::cout<<"matmul 128x128 took "<<std::chrono::duration<double, std::milli>(t1-t0).count()<<" ms";
#ifdef USE_NUMPY_CPP
    std::cout<<" (numpy-cpp blocked GEMM)";
#endif
    std::cout<<"\n";
    std::cout<<"C(0,0)="<<C(0,0)<<"\n";
    return 0;
}
