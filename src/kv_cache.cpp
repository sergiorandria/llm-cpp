#include "llm/kv_cache.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace llm {

// ─── CUDA helpers (gated behind USE_CUDA at compile time, runtime no-op) ───

#ifdef USE_CUDA
namespace {
bool cuda_kv_device_present() {
    static const bool present = [] {
        int n = 0;
        return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
    }();
    return present;
}
}  // namespace
#endif

void KVCache::cuda_alloc_layer(size_t layer) {
#ifdef USE_CUDA
    if (!cfg_.use_cuda || !cuda_kv_device_present()) return;
    size_t bytes = max_seq_len_ * n_embd_ * sizeof(float);
    void* dk = nullptr;
    void* dv = nullptr;
    if (cudaMalloc(&dk, bytes) != cudaSuccess) dk = nullptr;
    if (cudaMalloc(&dv, bytes) != cudaSuccess) dv = nullptr;
    // Zero-init so unwritten positions are safe
    if (dk) cudaMemset(dk, 0, bytes);
    if (dv) cudaMemset(dv, 0, bytes);
    gk_[layer] = dk;
    gv_[layer] = dv;
#else
    (void)layer;
#endif
}

void KVCache::cuda_free_all() {
#ifdef USE_CUDA
    for (auto& p : gk_) {
        if (p) cudaFree(p);
        p = nullptr;
    }
    for (auto& p : gv_) {
        if (p) cudaFree(p);
        p = nullptr;
    }
#endif
    gk_.clear();
    gv_.clear();
}

void KVCache::cuda_update(size_t layer, const Tensor& k, const Tensor& v) {
#ifdef USE_CUDA
    if (!cfg_.use_cuda || !cuda_kv_device_present()) return;
    if (layer >= gk_.size() || !gk_[layer]) return;
    // k, v are [tokens, n_embd] row-major on CPU. We store as [max_seq_len, n_embd] on GPU.
    // For each token i, write row at offset (cur_len_ + i).
    size_t n_tokens = std::min(k.shape[0], max_seq_len_ - cur_len_);
    if (n_tokens == 0) return;
    size_t row_bytes = n_embd_ * sizeof(float);
    // Contiguous row-major: copy each row to the correct GPU offset
    for (size_t i = 0; i < n_tokens; ++i) {
        size_t gpu_offset = (cur_len_ + i) * n_embd_;
        cudaMemcpy((float*)gk_[layer] + gpu_offset, k.data.data() + i * n_embd_, row_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy((float*)gv_[layer] + gpu_offset, v.data.data() + i * n_embd_, row_bytes,
                   cudaMemcpyHostToDevice);
    }
#else
    (void)layer;
    (void)k;
    (void)v;
#endif
}

void KVCache::cuda_get_slice(size_t layer, Tensor& out_k, Tensor& out_v) const {
#ifdef USE_CUDA
    if (!cfg_.use_cuda || !cuda_kv_device_present()) return;
    if (layer >= gk_.size() || !gk_[layer]) return;
    size_t n = std::min(cur_len_, max_seq_len_);
    if (n == 0) return;
    size_t row_bytes = n_embd_ * sizeof(float);
    size_t total_bytes = n * n_embd_ * sizeof(float);
    // GPU stores as [max_seq_len, n_embd], copy first n rows
    cudaMemcpy(out_k.data.data(), gk_[layer], total_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_v.data.data(), gv_[layer], total_bytes, cudaMemcpyDeviceToHost);
#else
    (void)layer;
    (void)out_k;
    (void)out_v;
#endif
}

// ─── Constructor / destructor / move ───

KVCache::KVCache(size_t n_layers, size_t max_seq_len, size_t n_embd, KVCacheConfig cfg)
    : n_layers_(n_layers), max_seq_len_(max_seq_len), n_embd_(n_embd), cfg_(cfg) {
    if (cfg_.page_size == 0) cfg_.page_size = 16;
    if (cfg_.paged) {
        pages_k_.assign(n_layers, {});
        pages_v_.assign(n_layers, {});
    } else if (cfg_.soa) {
        k_cache_.assign(n_layers, Tensor({n_embd, max_seq_len}, 0));
        v_cache_.assign(n_layers, Tensor({n_embd, max_seq_len}, 0));
    } else {
        k_cache_.assign(n_layers, Tensor({max_seq_len, n_embd}, 0));
        v_cache_.assign(n_layers, Tensor({max_seq_len, n_embd}, 0));
    }
    // GPU allocation: one [max_seq_len, n_embd] buffer per layer for K and V
    if (cfg_.use_cuda) {
        gk_.resize(n_layers, nullptr);
        gv_.resize(n_layers, nullptr);
        for (size_t l = 0; l < n_layers; ++l) cuda_alloc_layer(l);
    }
}

KVCache::~KVCache() {
    cuda_free_all();
}

KVCache::KVCache(KVCache&& o) noexcept
    : n_layers_(o.n_layers_),
      max_seq_len_(o.max_seq_len_),
      n_embd_(o.n_embd_),
      cfg_(o.cfg_),
      k_cache_(std::move(o.k_cache_)),
      v_cache_(std::move(o.v_cache_)),
      pages_k_(std::move(o.pages_k_)),
      pages_v_(std::move(o.pages_v_)),
      pool_(o.pool_),
      cur_len_(o.cur_len_),
      gk_(std::move(o.gk_)),
      gv_(std::move(o.gv_)) {
    o.gk_.clear();
    o.gv_.clear();
    o.pool_ = nullptr;
}

KVCache& KVCache::operator=(KVCache&& o) noexcept {
    if (this != &o) {
        cuda_free_all();
        n_layers_ = o.n_layers_;
        max_seq_len_ = o.max_seq_len_;
        n_embd_ = o.n_embd_;
        cfg_ = o.cfg_;
        k_cache_ = std::move(o.k_cache_);
        v_cache_ = std::move(o.v_cache_);
        pages_k_ = std::move(o.pages_k_);
        pages_v_ = std::move(o.pages_v_);
        pool_ = o.pool_;
        cur_len_ = o.cur_len_;
        gk_ = std::move(o.gk_);
        gv_ = std::move(o.gv_);
        o.gk_.clear();
        o.gv_.clear();
        o.pool_ = nullptr;
    }
    return *this;
}

// ─── CPU cache operations (unchanged logic) ───

void KVCache::ensure_page(size_t layer, size_t page) const {
    auto& pk = const_cast<std::vector<std::vector<Tensor>>&>(pages_k_);
    auto& pv = const_cast<std::vector<std::vector<Tensor>>&>(pages_v_);
    while (pk[layer].size() <= page) {
        if (pool_) {
            pk[layer].push_back(pool_->acquire({cfg_.page_size, n_embd_}));
            pv[layer].push_back(pool_->acquire({cfg_.page_size, n_embd_}));
        } else {
            pk[layer].emplace_back(Tensor({cfg_.page_size, n_embd_}, 0.0f));
            pv[layer].emplace_back(Tensor({cfg_.page_size, n_embd_}, 0.0f));
        }
    }
}

void KVCache::update(size_t layer, const Tensor& k, const Tensor& v) {
    if (layer >= n_layers_) return;
    // CPU cache update (always, so CPU fallback paths work)
    if (cfg_.paged) {
        for (size_t i = 0; i < k.shape[0] && cur_len_ + i < max_seq_len_; ++i) {
            size_t pos = cur_len_ + i, pg = pos / cfg_.page_size, off = pos % cfg_.page_size;
            ensure_page(layer, pg);
            for (size_t j = 0; j < n_embd_; ++j) pages_k_[layer][pg](off, j) = k(i, j);
        }
        for (size_t i = 0; i < v.shape[0] && cur_len_ + i < max_seq_len_; ++i) {
            size_t pos = cur_len_ + i, pg = pos / cfg_.page_size, off = pos % cfg_.page_size;
            ensure_page(layer, pg);
            for (size_t j = 0; j < n_embd_; ++j) pages_v_[layer][pg](off, j) = v(i, j);
        }
    } else if (layer < k_cache_.size()) {
        if (cfg_.soa) {
            for (size_t i = 0; i < k.shape[0] && cur_len_ + i < max_seq_len_; ++i)
                for (size_t j = 0; j < n_embd_; ++j) k_cache_[layer](j, cur_len_ + i) = k(i, j);
            for (size_t i = 0; i < v.shape[0] && cur_len_ + i < max_seq_len_; ++i)
                for (size_t j = 0; j < n_embd_; ++j) v_cache_[layer](j, cur_len_ + i) = v(i, j);
        } else {
            for (size_t i = 0; i < k.shape[0] && cur_len_ + i < max_seq_len_; ++i)
                for (size_t j = 0; j < n_embd_; ++j) k_cache_[layer](cur_len_ + i, j) = k(i, j);
            for (size_t i = 0; i < v.shape[0] && cur_len_ + i < max_seq_len_; ++i)
                for (size_t j = 0; j < n_embd_; ++j) v_cache_[layer](cur_len_ + i, j) = v(i, j);
        }
    }
    // GPU cache update (parallel path — bulk H2D per token row)
    cuda_update(layer, k, v);
}

Tensor KVCache::get_k(size_t layer) const {
    if (cfg_.paged) return get_k_slice(layer);
    return k_cache_[layer];
}

Tensor KVCache::get_v(size_t layer) const {
    if (cfg_.paged) return get_v_slice(layer);
    return v_cache_[layer];
}

Tensor KVCache::get_k_slice(size_t layer) const {
    size_t n = std::min(cur_len_, max_seq_len_);
    if (cfg_.paged) {
        Tensor out({n, n_embd_}, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            size_t pg = i / cfg_.page_size, off = i % cfg_.page_size;
            for (size_t j = 0; j < n_embd_; ++j) out(i, j) = pages_k_[layer][pg](off, j);
        }
        return out;
    }
    if (cfg_.soa) {
        Tensor out({n, n_embd_}, 0.0f);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n_embd_; ++j) out(i, j) = k_cache_[layer](j, i);
        return out;
    }
    Tensor out({n, n_embd_}, 0.0f);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n_embd_; ++j) out(i, j) = k_cache_[layer](i, j);
    return out;
}

Tensor KVCache::get_v_slice(size_t layer) const {
    size_t n = std::min(cur_len_, max_seq_len_);
    if (cfg_.paged) {
        Tensor out({n, n_embd_}, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            size_t pg = i / cfg_.page_size, off = i % cfg_.page_size;
            for (size_t j = 0; j < n_embd_; ++j) out(i, j) = pages_v_[layer][pg](off, j);
        }
        return out;
    }
    if (cfg_.soa) {
        Tensor out({n, n_embd_}, 0.0f);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n_embd_; ++j) out(i, j) = v_cache_[layer](j, i);
        return out;
    }
    Tensor out({n, n_embd_}, 0.0f);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n_embd_; ++j) out(i, j) = v_cache_[layer](i, j);
    return out;
}

void KVCache::clear() {
    cur_len_ = 0;
    for (auto& t : k_cache_) t.fill(0);
    for (auto& t : v_cache_) t.fill(0);
    // Zero GPU memory
#ifdef USE_CUDA
    if (cfg_.use_cuda) {
        size_t bytes = max_seq_len_ * n_embd_ * sizeof(float);
        for (size_t l = 0; l < n_layers_; ++l) {
            if (gk_[l]) cudaMemset(gk_[l], 0, bytes);
            if (gv_[l]) cudaMemset(gv_[l], 0, bytes);
        }
    }
#endif
    evict();
}

void KVCache::evict() {
    // I85: with a pool, pages go back for reuse (free-list); else freed
    if (pool_) {
        for (auto& pl : pages_k_)
            for (auto& pg : pl) pool_->release(std::move(pg));
        for (auto& pl : pages_v_)
            for (auto& pg : pl) pool_->release(std::move(pg));
    }
    for (auto& pl : pages_k_) pl.clear();
    for (auto& pl : pages_v_) pl.clear();
}

size_t KVCache::num_pages(size_t layer) const {
    if (!cfg_.paged || layer >= pages_k_.size()) return 0;
    return pages_k_[layer].size();
}

}  // namespace llm
