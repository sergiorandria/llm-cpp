#include "llm/gguf.h"
#include <iostream>
namespace llm {
bool save_gguf(const GPT& model, const std::string& path){ std::cout<<"[gguf] save stub "<<path<<"\n"; (void)model; return true; }
bool load_gguf(GPT& model, const std::string& path){ std::cout<<"[gguf] load stub "<<path<<"\n"; (void)model; return true; }
}
