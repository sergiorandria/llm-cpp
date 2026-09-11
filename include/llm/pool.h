#pragma once
// I86: Tensor memory pool — shape-keyed recycling of activation-sized Tensors.
// I90: pooled raw scratch is 64B-aligned for SIMD.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>
namespace llm {
// 64B-aligned malloc/free pair (I90)
inline void* aligned_malloc64(size_t bytes) {
    if (bytes == 0) bytes = 64;
    size_t padded = (bytes + 63) & ~(size_t)63;
    void* p = std::aligned_alloc(64, padded);
    return p;
}
inline void aligned_free64(void* p) { std::free(p); }

// Aligned STL allocator for Tensor storage (I90: every data.data() 64B-aligned)
template <typename T, size_t Align = 64>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Align>&) {}
    T* allocate(size_t n) {
        if (n == 0) return nullptr;
        size_t bytes = n * sizeof(T);
        size_t padded = (bytes + Align - 1) & ~(Align - 1);
        void* p = std::aligned_alloc(Align, padded);
        if (!p) throw std::bad_alloc();
        return (T*)p;
    }
    void deallocate(T* p, size_t) noexcept { std::free(p); }
    template <typename U> struct rebind { using other = AlignedAllocator<U, Align>; };
};
template <typename T, size_t A> bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<T, A>&) { return true; }
template <typename T, size_t A> bool operator!=(const AlignedAllocator<T, A>& a, const AlignedAllocator<T, A>& b) { return !(a == b); }

// Aligned float storage shared by Tensor::data/grad (I90)
using FloatVec = std::vector<float, AlignedAllocator<float, 64>>;

inline bool is_aligned64(const void* p) { return ((uintptr_t)p & 63) == 0; }

// I86: shape-keyed Tensor recycling pool (mutex-guarded; stats for tests).
// Acquire reuses a cached Tensor of equal numel (reshaped); release returns it.
// Backed by 64B-aligned Tensor storage (I90), so pooled pages stay aligned.
class Tensor;
class TensorPool {
public:
    Tensor acquire(const std::vector<size_t>& shape);
    void release(Tensor&& t);
    size_t hits() const { return hits_.load(); }
    size_t misses() const { return misses_.load(); }
    size_t cached() const;
private:
    static size_t key(const std::vector<size_t>& shape);
    mutable std::mutex mu_;
    std::unordered_map<size_t, std::vector<Tensor>> free_;
    std::atomic<size_t> hits_{0}, misses_{0};
};
} // namespace llm
