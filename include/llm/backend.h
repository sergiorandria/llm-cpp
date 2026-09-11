#pragma once
// I87: compute-backend abstraction. CPU is always available; CUDA is selected
// with -DUSE_CUDA=ON when CUDAToolkit is found, otherwise the build warns and
// stays on CPU. Tensor::matmul dispatches on active_backend().
namespace llm {
enum class Backend { CPU, CUDA };
Backend active_backend();
const char* backend_name();
bool cuda_available();
}  // namespace llm
