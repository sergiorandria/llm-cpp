#include "llm/prefetch.h"
namespace llm {
PrefetchLoader::PrefetchLoader(const Dataset& ds, size_t batch_size, size_t prefetch)
    : ds_(ds), batch_size_(batch_size), prefetch_(prefetch) {}
std::vector<std::vector<int>> PrefetchLoader::next(){
    if(cursor_ >= ds_.size()) return {};
    auto b = ds_.get_batch(cursor_, batch_size_);
    cursor_ += batch_size_;
    return {b};
}
bool PrefetchLoader::has_next() const { return cursor_ < ds_.size(); }
}
