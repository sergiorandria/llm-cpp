#include "llm/prefetch.h"

namespace llm {

PrefetchLoader::PrefetchLoader(const Dataset& ds, size_t batch_size, size_t prefetch)
    : ds_(ds), batch_size_(batch_size), prefetch_(prefetch) {
    if(prefetch_ > 0){
        worker_ = std::thread(&PrefetchLoader::worker_loop, this);
    }
}

PrefetchLoader::~PrefetchLoader(){
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
    }
    cv_producer_.notify_all();
    cv_consumer_.notify_all();
    if(worker_.joinable()) worker_.join();
}

void PrefetchLoader::worker_loop(){
    while(true){
        std::unique_lock<std::mutex> lock(mtx_);
        // Wait until queue has space or stop
        cv_producer_.wait(lock, [&]{ return stop_ || queue_.size() < prefetch_ || done_; });
        if(stop_) break;
        if(done_) break;
        if(cursor_ >= ds_.size()){
            done_ = true;
            cv_consumer_.notify_all();
            break;
        }
        // Reserve slot: check again size
        if(queue_.size() >= prefetch_){
            continue;
        }
        size_t cur = cursor_;
        cursor_ += batch_size_;
        if(cur >= ds_.size()){
            done_ = true;
            cv_consumer_.notify_all();
            break;
        }
        lock.unlock();
        // Do I/O outside lock
        auto batch = ds_.get_batch(cur, batch_size_);
        std::vector<std::vector<int>> wrapped = {batch};
        lock.lock();
        queue_.push(std::move(wrapped));
        cv_consumer_.notify_one();
        if(cursor_ >= ds_.size()){
            done_ = true;
            cv_consumer_.notify_all();
        }
    }
}

std::vector<std::vector<int>> PrefetchLoader::next(){
    if(prefetch_==0){
        // Synchronous fallback when prefetch==0
        std::lock_guard<std::mutex> lock(mtx_);
        if(cursor_ >= ds_.size()) return {};
        auto b = ds_.get_batch(cursor_, batch_size_);
        cursor_ += batch_size_;
        return {b};
    }
    std::unique_lock<std::mutex> lock(mtx_);
    cv_consumer_.wait(lock, [&]{ return !queue_.empty() || done_; });
    if(queue_.empty()){
        return {};
    }
    auto batch = std::move(queue_.front());
    queue_.pop();
    cv_producer_.notify_one();
    return batch;
}

bool PrefetchLoader::has_next() const {
    std::lock_guard<std::mutex> lock(mtx_);
    if(!queue_.empty()) return true;
    return cursor_ < ds_.size() || !queue_.empty();
}

}
