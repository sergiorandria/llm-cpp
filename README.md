# llm-cpp

![Build](https://img.shields.io/badge/build-passing-brightgreen) ![C++17](https://img.shields.io/badge/C%2B%2B-17-blue) ![License](https://img.shields.io/badge/license-MIT-green)

Implementation of a Large Language Model (LLM) from scratch in C++.

> No frameworks — just C++17, linear algebra, and Transformer architecture.

## Features (planned)

- [x] Tensor & autograd primitives (CPU)
- [ ] BPE Tokenizer
- [ ] Multi-Head Self-Attention
- [ ] Transformer Block (Attention + FFN + LayerNorm + Residuals)
- [ ] Autoregressive LLM (GPT-style)
- [ ] Training loop (AdamW, cross-entropy, gradient clipping)
- [ ] Inference with KV-cache & sampling (temperature, top-k, top-p)
- [ ] GGUF/Checkpoint save & load
- [ ] Minimal dataset loader (TinyStories / Shakespeare)

## Quick Start

### Requirements

- CMake >= 3.16
- C++17 compiler (g++ >= 9 / clang++ >= 10)
- Make / Ninja

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/llm-cpp --help
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
