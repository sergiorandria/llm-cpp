# Numpy-CPP Backend

`llm-cpp` now supports `numpy-cpp` (`/home/sergio/Project/numpy-cpp`) as an accelerated third-party backend.

## Integration

`CMakeLists.txt:20` — C++20, `add_subdirectory(/home/sergio/Project/numpy-cpp)` when `USE_NUMPY_CPP=ON` (default). Fetches via `FetchContent` if local checkout missing.

Link: `target_link_libraries(llm-cpp PRIVATE numpy-cpp)` (`numpy-cpp` or `numpy-cpp::numpy-cpp` alias), defines `USE_NUMPY_CPP=1`.

## Tensor Interop — `include/llm/tensor.h:58` / `src/tensor.cpp:44`

- `Tensor::to_ndarray()` / `from_ndarray()` — copy between `std::vector<float>` + `shape: vector<size_t>` and `np::ndarray<float>` (`shape: vector<int>`).
- `matmul()` `src/tensor.cpp:120` — dispatches to `np::linalg::matmul` (blocked GEMM `BLOCK=32`, SSE4.2/AVX2 dispatched, `parallel_for` >4096).
- `randn()` `src/tensor.cpp:34` — via `np::random::Generator(seed).standard_normal<float>(shape)` (PCG64 + Box-Muller) then `*std + mean`.
- `transpose()` `src/tensor.cpp:105` — `np::ndarray::transpose()` view (strides, shared storage, no memcpy).
- `numpy_utils.h` — helpers: `mean_np`, `gelu_np`, `softmax_np` (wrappers for `np::mean`, `np::exp` SIMD).

## Attention / Transformer

`MultiHeadAttention::forward` and `TransformerBlock::forward` benefit automatically because they call `Tensor::matmul`/`layernorm`. No code change needed — `llm::Tensor` is the acceleration point.

Future: `src/attention.cpp` can directly use `np::linalg::matmul` for QK^T and add `np::exp` for softmax to exploit `np::simd.hpp`.

## Example — `examples/numpy_backend.cpp:1`

```cpp
#include "llm/tensor.h"
auto C = A.matmul(B); // -> np::linalg::matmul
auto np_c = np::linalg::matmul(A.to_ndarray(), B.to_ndarray());
llm::Tensor W({4,4}); W.randn(0,0.02f); // -> np::random::Generator
```

Run: `./build/tests/test_numpy_backend` — verifies `C == np_c`, `mean`, etc. `ctest 5/5`.

## Performance

- `numpy-cpp` fast paths are `[[likely]]`-guarded: `is_contiguous() -> ptr[i]` direct, `copyto` via `memcpy` when contiguous + `__restrict`.
- `matmul`: blocked `32` + `parallel_for` threshold `4096` + SIMD (`simd.hpp:983`).
- `randn`: PCG64 (same as NumPy) vs `std::mt19937`.

Benchmark: `examples/numpy_backend` prints `W randn` with PCG64 determinism; `scripts/benchmark.py` can compare `USE_NUMPY_CPP=ON/OFF`.

## Build Variants

- Release + AVX2 + LTO + Threading: `cmake -DNP_ENABLE_AVX2=ON -DNP_ENABLE_LTO=ON -DNP_USE_THREADING=ON` (passed through to numpy-cpp subdir).
- Disable: `-DUSE_NUMPY_CPP=OFF` falls back to naive `O(n^3)` loop + `std::mt19937`.

## References

- numpy-cpp: https://github.com/sergiorandria/numpy-cpp — header-only, C++20, 760+ NumPy 2.2 routines.
- llm-cpp Tensor: `include/llm/tensor.h:11` — now `#ifdef USE_NUMPY_CPP` includes `<np/np.hpp>`.
