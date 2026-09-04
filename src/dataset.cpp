#include "llm/dataset.h"
#include "llm/tokenizer.h"
#include <fstream>
#include <random>
namespace llm {
Dataset::Dataset(const std::string& path, size_t block_size): block_size_(block_size){
    std::ifstream in(path);
    std::string text((std::istreambuf_iterator<char>(in)), {});
    if(text.empty()) text="hello world\n";
    Tokenizer tok;
    tokens_=tok.encode(text);
}
std::vector<int> Dataset::get_batch(size_t idx, size_t batch_size) const {
    std::vector<int> batch;
    for(size_t i=0;i<batch_size && idx+i<tokens_.size(); ++i) batch.push_back(tokens_[idx+i]);
    return batch;
}
DataLoader::DataLoader(const Dataset& ds, size_t batch_size, bool shuffle): ds_(ds), batch_size_(batch_size), shuffle_(shuffle){}
std::vector<std::vector<int>> DataLoader::next_batch(){
    std::vector<std::vector<int>> b;
    if(cursor_ >= ds_.size()) return b;
    b.push_back(ds_.get_batch(cursor_, batch_size_));
    cursor_+=batch_size_;
    return b;
}
bool DataLoader::has_next() const { return cursor_ < ds_.size(); }
void DataLoader::reset(){ cursor_=0; }
}
