# llm-cpp

![Build](https://img.shields.io/badge/build-passing-brightgreen) ![C++20](https://img.shields.io/badge/C%2B%2B-20-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![numpy-cpp](https://img.shields.io/badge/backend-numpy--cpp%20SIMD-orange)

Implementation of a Large Language Model (LLM) from scratch in C++.

> No frameworks — just C++20, `numpy-cpp` (SIMD, blocked GEMM, PCG64), and Transformer architecture.

## Features

- [x] Tensor primitives (CPU, numpy-cpp GEMM/SIMD when enabled) + manual backward (matmul grad, layernorm backward)
- [x] BPE Tokenizer (char-level + BPE merges, train/encode/decode)
- [x] Multi-Head Self-Attention (causal, per-head split, bias applied when enabled, manual backward)
- [x] Transformer Block (Attention + FFN + LayerNorm + Residuals, SwiGLU) — full manual backward
- [x] Autoregressive LLM (GPT-style, RoPE/sinusoidal/learned pos, weight tying) — forward_with_hidden + backward
- [x] Training loop (real backpropagation through TransformerBlocks to embeddings; random-noise fallback gated behind `--allow-untrained-params`)
- [x] Inference sampling (temperature, top-k, top-p, repetition penalty) — **per-layer KV-cache O(n)** via `KVCache` + `forward_incremental` (`attention.h:15`, `transformer.h:13`, `model.cpp:156` incremental, `test_kv_cache` + `test_model` green)
- [x] FlashAttention tiled incremental (`flash_attention.h:9` `flash_attention_incremental` 1×Hd over pos+1, block `FLASH_BLOCK_SIZE` autotuned via `scripts/autotune_flash.py` → `flash_config.h:3` `32`, `bench_matmul` dashboard `docs/BENCHMARK_DASHBOARD.md`)
- [x] KV-cache SoA (`kv_cache.h:5` `KVCacheConfig{soa}` `[n_embd, max_seq_len]` vs AoS, `test_kv_cache_per_layer` SoA/AoS equivalence)
- [x] Tensor matmul OpenBLAS (`tensor.cpp:100` `cblas_sgemm` RowMajor when `-DUSE_OPENBLAS=ON`, `cmake/FindOpenBLAS.cmake:1` stub→real, fallback blocked GEMM)
- [x] Checkpoint save & load (binary v3: header + all tensors incl. TransformerBlocks)
- [x] GGUF save & load (`save_gguf`/`load_gguf` — GGUF v3 32-byte aligned: magic `"GGUF"` + KV metadata + tensor info + data, roundtrip bit-exact via `test_gguf_roundtrip`)
- [x] Speculative decoding (`SpeculativeDecoder` k=4 — draft proposes k tokens, target verifies in parallel via single forward pass over draft sequence, accepts longest matching prefix, bonus token; greedy equivalence via `test_speculative`)
- [x] Minimal dataset loader (TinyStories / Shakespeare) — via Dataset/DataLoader
- [x] ALiBi bias (`alibi_bias`) — linear length extrapolation
- [x] Sliding window attention (512 window, 32k via `sliding_attention`)
- [x] MQA (`mqa_forward`, single KV head, 8× KV-cache reduction)
- [x] Early exit (`should_early_exit`, softmax max threshold)
- [x] Evaluation harness (`evaluate` perplexity real via `compute_loss`, `mmlu`/`hellaswag` token-accuracy proxies, not stubbed to 0 — `test_eval` asserts ppl>0 and accuracy in [0,1])
- [x] Quantization — int8 (`quantize_int8` with scale) + 4-bit GPTQ (`quantize_4bit` now outputs per-group `scale` via `Tensor& scale_out` overload, `dequantize_4bit` uses returned scale, reconstruction error <0.5 — `test_gptq` roundtrip)
- [x] Pruning (`prune_model` magnitude, `sparsity`)
- [x] Distillation (`distill_step` KL, temp=2.0, requires real grads)
- [x] MoE (`MoEFFN` 8 experts top-2 + load balancing)
- [x] RLHF (`RewardModel` learned linear head `n_embd->1` via `config().n_embd`, `train_step` Bradley-Terry pairwise logistic loss, `ppo_step` PPO clipped ratio `min(ratio*A, clip(ratio)*A)`, advantage baseline, `kl_coef` KL penalty vs old policy, Adam optimizer)
- [x] Pipeline parallelism (`Pipeline::train_step` micro-batch 4, staged `n_layers/stages` validated, sequential accumulation with Adam — honest single-process staged, `GradientAccumulator` alias)
- [x] Beam search (`beam_search` tracks cumulative logprob, expands top-k per beam, keeps best — `test_beam_search` asserts score ≥ greedy)
- [x] Fused ops (`fused_layernorm_residual` single-pass fused + `fused_gelu`/`fused_matmul_gelu` single-pass OpenMP/SIMD, not alias)
- [x] Prefetch (`PrefetchLoader` background thread producer/consumer queue, `prefetch` batches ahead, `has_next`/`next` thread-safe)

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
ctest --test-dir build  # 31/31 including kv_cache_per_layer
./build/tests/test_numpy_backend  # demo: matmul, randn, transpose via np
# Autotune Flash block
python3 scripts/autotune_flash.py  # → include/llm/flash_config.h + docs/BENCHMARK_DASHBOARD.md
# Optional OpenBLAS
cmake -B build -DUSE_OPENBLAS=ON  # cblas_sgemm in tensor.cpp:100

# Disable numpy-cpp (fallback naive CPU)
cmake -B build -DUSE_NUMPY_CPP=OFF
```

### Train (example)

```bash
./build/llm-cpp train --config config/config.json --data data/input.txt
```

### Generate

```bash
./build/llm-cpp generate --prompt "Hello, world" --max_tokens 20
./build/llm-cpp generate --config config/config.json.example --checkpoint checkpoints/model.bin --prompt "Hello" --temperature 0.8 --top_k 40 --top_p 0.9
# --checkpoint now loads binary v3 (all blocks) when config matches; mismatched config warns and uses random init
# KV-cache is per-layer (`KVCache` `kv_cache.h:5`) — O(n) generation via `forward_incremental` (was O(n²) unified recompute)
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
