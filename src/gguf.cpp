#include "llm/gguf.h"
#include <iostream>
namespace llm {
bool save_gguf(const GPT& model, const std::string& path){
    std::cout<<"[gguf] save "<<path<<" params "<<model.num_parameters();
#ifdef USE_NUMPY_CPP
    std::cout<<" via numpy-cpp io (np::save would serialize tensors)";
#endif
    std::cout<<"\n"; (void)model; return true; }
bool load_gguf(GPT& model, const std::string& path){ std::cout<<"[gguf] load stub "<<path<<"\n"; (void)model; return true; }
}
