# llm-cpp

![Build](https://img.shields.io/badge/build-passing-brightgreen) ![C++20](https://img.shields.io/badge/C%2B%2B-20-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![numpy-cpp](https://img.shields.io/badge/backend-numpy--cpp%20SIMD-orange)

Implementation of a Large Language Model (LLM) from scratch in C++.

> No frameworks — just C++20, `numpy-cpp` (SIMD, blocked GEMM, PCG64), and Transformer architecture.

## Features (planned)

- [x] Tensor & autograd primitives (CPU)
- [x] BPE Tokenizer (char-level + BPE merges)
- [x] Multi-Head Self-Attention (causal, KV-cache stub)
- [x] Transformer Block (Attention + FFN + LayerNorm + Residuals)
- [x] Autoregressive LLM (GPT-style)
- [x] Training loop (AdamW, cross-entropy, gradient clipping)
- [x] Inference with KV-cache & sampling (temperature, top-k, top-p)
- [x] Checkpoint save & load (binary v1)
- [ ] Minimal dataset loader (TinyStories / Shakespeare)

## Quick Start

### Requirements

- CMake >= 3.20
- C++20 compiler (g++ >= 11 / clang++ >= 15)
- Make / Ninja
- Optional: [numpy-cpp](https://github.com/sergiorandria/numpy-cpp) (auto-fetched, or local at `/home/sergio/Project/numpy-cpp`) — provides SIMD (SSE4.2/AVX2/NEON), blocked GEMM, PCG64

### Build

```bash
# With numpy-cpp accelerated backend (default, C++20 + SIMD + OpenMP)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_NUMPY_CPP=ON -DBUILD_TESTS=ON
cmake --build build -j
./build/llm-cpp --help
ctest --test-dir build  # 5/5 including numpy_backend
./build/tests/test_numpy_backend  # demo: matmul, randn, transpose via np

# Disable numpy-cpp (fallback naive CPU)
cmake -B build -DUSE_NUMPY_CPP=OFF
```

### Train (example)

```bash
./build/llm-cpp train --config config/config.json --data data/input.txt
```

### Generate

```bash
./build/llm-cpp generate --checkpoint build/model.bin --prompt "Hello, world"
```

## Project Structure

```
.
├── CMakeLists.txt
├── include/llm/        # Headers: tensor, attention, transformer, model
├── src/                # Implementations
├── config/             # Example model configs
├── data/               # Training data (not tracked)
└── build/              # Build artifacts (not tracked)
```

## Architecture

Standard decoder-only Transformer:

```
Tokens -> TokenEmbedding + PositionalEmbedding -> N x TransformerBlock -> LayerNorm -> LM Head -> Logits
```

TransformerBlock:
```
x -> LayerNorm -> MultiHeadAttention -> Residual -> LayerNorm -> FeedForward -> Residual
```

## Roadmap

Progress: 100 tasks completed on 2026-09-04 — see docs/CHANGELOG.md

1.  Core Tensor ops (matmul, softmax, layernorm)
2.  Tokenizer (BPE)
3.  Single Transformer block forward pass
4.  Full model + training on tiny dataset
5.  Optimizations (KV-cache, quantization)

## Contributing

Collaborators: @Mokabe000

PRs welcome!

## License

MIT
