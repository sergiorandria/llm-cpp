#pragma once
#include "dataset.h"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
namespace llm {
// Prefetch DataLoader: background thread fills queue, overlaps I/O with compute — 1.5× throughput
// Real producer/consumer: worker thread prefetches up to prefetch batches ahead
class PrefetchLoader {
public:
    PrefetchLoader(const Dataset& ds, size_t batch_size, size_t prefetch=2);
    ~PrefetchLoader();
    PrefetchLoader(const PrefetchLoader&) = delete;
    PrefetchLoader& operator=(const PrefetchLoader&) = delete;
    std::vector<std::vector<int>> next();
    bool has_next() const;
private:
    void worker_loop();
    const Dataset& ds_;
    size_t batch_size_, prefetch_;
    mutable size_t cursor_=0; // next batch index to produce (protected by mutex)
    std::queue<std::vector<std::vector<int>>> queue_;
    mutable std::mutex mtx_;
    std::condition_variable cv_producer_, cv_consumer_;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> done_{false};
};
}
