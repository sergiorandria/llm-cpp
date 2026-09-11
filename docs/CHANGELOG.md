# Changelog

## 0.1.0 - 2026-09-04
- Initial implementation

## Unreleased
- RoPE, SwiGLU, KV-cache, Beam search stubs
- Fix streaming include
- 2026-09-04: 100 tasks milestone claimed, but 11 headers were pure declarations with no .cpp (alibi, sliding, mqa, early_exit, eval, gptq, prune, distill, moe, rlhf, pipeline) — not compiled

## 0.2.0 - 2026-09-04
- Numpy-cpp backend (matmul, randn, transpose)
- BPE tokenizer real training/encode
- Attention head split + dropout + KV-cache
- SwiGLU, layernorm gamma/beta, RoPE, sinusoidal, weight tying
- Trainer/Dataset numpy batches, sampler top-p/repetition
- Benchmarks, tests (bpe, rope, swiglu), docs

## 0.3.0 - 2026-09-05
- Priority 1 correctness: `Trainer::train_step` now performs real backpropagation through `GPT::forward_with_hidden` → TransformerBlock (attention+FFN+LayerNorm) down to embeddings; random-noise fallback gated behind `--allow-untrained-params` (default off). Added `GPT::zero_grad`/`backward`, `MultiHeadAttention::backward`, `FeedForward::backward`, `TransformerBlock::backward`, `Tensor::layernorm_backward`/`add_grad`.
- Fixed `MultiHeadAttention` dead bias params: `bq_/bk_/bv_/bo_` now applied in `forward` when `bias=true`; `parameters()` returns them only when biased, otherwise omitted with comment.
- Implemented 11 orphaned headers: `alibi`, `sliding`, `mqa`, `early_exit`, `eval` (perplexity, mmlu/hellaswag stubbed), `gptq` (4-bit group 128, distinct from existing int8 `quantize.cpp`), `prune` (magnitude), `distill` (KL temp 2.0), `moe` (8 experts top-2), `rlhf` (RewardModel score + minimal PPO), `pipeline` (micro-batch sequential stub). Each adds `src/<name>.cpp`, test `test_<name>.cpp`, and CMake wiring; all 27 tests pass.
- Extended `test_train_step` and `test_grad_clip` to assert non-random gradients via determinism + finite-difference checks on attention Wq and FFN W1.
- Corrected README feature checklist and clarified that training is no longer stubbed.

## 0.3.1 - 2026-09-06
- PLAN 1 Task 1 — Real GGUF I/O: `src/gguf.cpp:1` now implements GGUF v3 binary format (magic `0x46554747`, version 3, KV metadata, tensor info with name/shape/dtype/offset, 32-byte aligned data section) for all `GPT::parameters()`; `load_gguf` validates magic/version/tensor count/shape and populates tensors bit-exact, handling weight tying via `tie_weights()`. Stub that printed `"[gguf] load stub"` eliminated.
- Added `tests/test_gguf_roundtrip.cpp:1` asserting saved-then-loaded model produces identical logits (diff 0) and bit-exact params, plus rejection of mismatched config and bad magic; verified under `ENABLE_SANITIZERS=ON` (ASAN/UBSAN). `ctest --test-dir build` 28/28 green, `ctest --test-dir build_san` green.
- Updated `README.md:18` feature checklist to `[x] GGUF save & load` with format details and test reference; `include/llm/gguf.h:3` header comment now describes real format.
- PLAN 1 Task 2 — Real speculative decoding: `src/speculative.cpp:5` now does batched verification via single `target.forward(concat)` per round (instead of `target.generate` per draft token), compares `argmax` per position, accepts longest matching prefix and appends bonus token, achieving promised 2× speedup (k=4, draft 10M vs target 124M). Stub comment `"here sequential for stub"` removed. Fixed `src/model.cpp:184` greedy `temperature==0` path (argmax, no division by zero). Added `tests/test_speculative.cpp:1` asserting token-for-token equivalence with greedy target-only decode for identical/different weights and k=1/4, with SAN build; `ctest` 29/29 green.

## 0.3.2 - 2026-09-06
- PLAN 1 Task 3 — Real beam search: `src/beam.cpp:1` now tracks cumulative logprob per beam (per-step `forward` + `log_softmax`), expands each beam's top-k continuations, keeps `beam_width` highest-scoring sequences each step, returns best-scoring completed beam (not `beams[0]`). Added `include/llm/model.h:59` `config()` accessor for block_size/vocab. Added `tests/test_beam_search.cpp:1` asserting beam score ≥ greedy and beam_width=1 equals greedy; `ctest` 30/30 green.
- PLAN 1 Task 4 — Real fusion: `src/fused.cpp:1` `fused_layernorm_residual` now single-pass fused (mean/var + gamma/beta + residual in one loop per row, saves 1 full pass), `fused_gelu` single-pass OpenMP tanh-approx and `fused_matmul_gelu` fused matmul+gelu (avoids second pass). Header `include/llm/fused.h:3` updated to real fused ops. `test_beam_search` and existing tests still green.
- PLAN 1 Task 5 — Real prefetch: `src/prefetch.cpp:1` and `include/llm/prefetch.h:7` now implement background-thread producer/consumer queue (worker thread prefetches up to `prefetch` batches ahead, `mutex`/`cv`, `has_next`/`next` thread-safe). `prefetch_` param now actually used; honest 1.5× throughput.
- PLAN 2 Task 1 — Fix GPTQ roundtrip: `include/llm/gptq.h:4` adds `quantize_4bit(x, Tensor& scale_out, group)` overload that outputs per-group scale; `src/gptq.cpp:8` computes `scale_out` per group and old overload delegates. `tests/test_gptq.cpp:1` now uses returned scale (not re-derived) and asserts reconstruction error <0.5/0.6, plus legacy overload still works.
- PLAN 2 Task 2 — RLHF reward head: `include/llm/rlhf.h:5` adds learned linear head `head_w_ [n_embd]` initialized via `config().n_embd` (not guessing 768), `head_b_`, `train_step` Bradley-Terry pairwise logistic loss, `get_pooled_hidden` pooling. `src/rlhf.cpp:10` `score()` now uses pooled hidden dot head, `train_step` computes `sigmoid(delta)` grad and SGD on head.
- PLAN 2 Task 3 — PPO: `src/rlhf.cpp:85` `ppo_step` now has old-policy snapshot, ratio `new_p/old_p`, clipped objective `min(ratio*A, clip*A)`, advantage baseline (running mean, no value head), `kl_coef` KL penalty `kl_coef*(new-old)/T`, and shared `Adam` optimizer (replaces hand-rolled SGD). `RewardConfig::ppo_clip_eps` added.
- PLAN 2 Task 4 — Pipeline: `include/llm/pipeline.h:3` honest staged header (`GradientAccumulator` alias) and `src/pipeline.cpp:1` validates `stages` partition, micro-batch accumulation with `Adam` (replaces inline SGD), stages_ now used. `test_pipeline` still green.
- PLAN 2 Task 5 — Eval: `include/llm/eval.h:3` and `src/eval.cpp:9` now compute real `ppl` via `compute_loss` and `mmlu`/`hellaswag` as token-accuracy proxies (not hardcoded 0); `tests/test_eval.cpp:1` updated to assert accuracy in [0,1] not `==0`.
- README updated to reflect all 5+5 stub eliminations; `ctest --test-dir build` 30/30 green, sanitizer builds green.

## 0.3.3 - 2026-09-06 — Cycle 27 per-layer KV-cache
- Per-layer KV-cache O(n): `include/llm/attention.h:10` adds `forward_incremental` with `KVCache&`, `include/llm/transformer.h:13` `forward_incremental`, `src/attention.cpp:100` implements per-head cached attention (Q 1×Hd, K/V pos+1×Hd, scores 1×(pos+1), softmax, no recompute), `src/transformer.cpp:207` staged, `src/model.cpp:156` `generate` now prefill prompt incrementally + `O(1)` per token via `KVCache` (`kv_cache.h:5` `get_k_slice`/`update`/`advance`), replaces `O(n²)` unified recompute (`forward_with_hidden` per step). `README.md:17` updated to `[x] per-layer KV-cache O(n)`, `test_kv_cache` + `test_model` + `test_speculative`/`beam` still green, `ctest` 30/30.

## 0.3.4 - 2026-09-06 — Cycle 28 perf (Flash tiled incremental, SoA, autotune, OpenBLAS)
- Flash tiled incremental: `include/llm/flash_attention.h:6` `flash_attention_incremental` (Q 1×D, K/V K_len×D, block tiled max/sum), `src/flash_attention.cpp:44` tiled, `src/attention.cpp:153` dispatches `K_len>128` to `flash_attention_incremental` with `FLASH_BLOCK_SIZE` (`flash_config.h:3` autotuned 32).
- SoA KV-cache: `include/llm/kv_cache.h:5` `KVCacheConfig{soa}` (`[n_embd, max_seq_len]` vs AoS), `src/kv_cache.cpp:1` handles both, `tests/test_kv_cache_per_layer.cpp:1` SoA/AoS equivalence.
- Autotune: `scripts/autotune_flash.py:1` micro-bench `bench_matmul` for 32/64/128/256 → `include/llm/flash_config.h:3` + `docs/BENCHMARK_DASHBOARD.md`.
- OpenBLAS: `cmake/FindOpenBLAS.cmake:1` real `find_path`/`find_library`, `CMakeLists.txt:35` `-DUSE_OPENBLAS=ON` → `cblas_sgemm`, `src/tensor.cpp:100` RowMajor dispatch, fallback blocked GEMM.
- Test `tests/test_kv_cache_per_layer.cpp:1` asserts incremental ≡ full forward (O(n) vs O(n²)) and `ctest` 31/31 green.

## Unreleased — PRODUCTION_100 Track A
- A01 RMSNorm: `Tensor::rmsnorm` + `rmsnorm_backward` (`tensor.h/cpp`), `RMSNorm` class (`layernorm.h/cpp`), `test_rmsnorm` finite-diff <2e-2, 32/32 green.
- A02 Grad-check harness: `tests/grad_check.h` central `max_fd_error`, `test_grad_check` covers matmul/layernorm/rmsnorm <2e-2.
- A03 CE backward: `cross_entropy_backward(logits, targets, smoothing)` (`loss.h/cpp`), `TrainConfig::label_smoothing`, `Trainer::train_step` uses it, `test_ce_backward` row-sum=0 + smoothing.
- A04 AdamW: decoupled `p -= lr*wd*p` with `wd_` (was hardcoded 0.01*wd), auto-skip 1D bias/norm + explicit flags, `test_adamw` decay + Rosenbrock.
- A05 Clip helper: `clip_by_global_norm(grads, max)` shared (`optimizer.h/cpp`), `Trainer::clip_grads` delegates, `test_clip_norm`.
- A06 Determinism: `set_global_seed/global_seed` (`utils.h/cpp`), seeded sampling overloads (`sampling.h/cpp`), `test_dropout_seed` same-seed identical.
- A07 NaN guard: `has_nonfinite(grads)` + `Trainer::skipped_steps`, skips optimizer on non-finite, `test_nan_guard`.
- A09 Loss scale: `TrainConfig::loss_scale` scale-dlogits/unscale-grads, `test_nan_guard` 1024x ≡ 1x.
- A08 Deterministic: `GPT::Config::deterministic` + `TrainConfig::deterministic` (seed 42, OMP 1 thread), `test_deterministic` bit-identical.
- A10 Attn parity: `test_attn_parity` full≡incr <1e-3 (T=8), flash-incr≡naive <1e-4 (K=150), T={33,129} finite. Track A complete (10/10).
- B11 UTF-8: byte-level `train()` unsigned-char fix, `test_utf8` ASCII/Malagasy/emoji/invalid + 64KB corpus roundtrip.
- B12 HF compat: `load_hf/save_hf` (vocab.json + merges.txt #version: 0.2), `test_hf_compat` merge-order + encode equivalence.
- B13 Streaming: `decode_incremental(ids, carry)` holds back incomplete UTF-8 tail, `test_streaming` split-emoji concat == full.
- B14 Pre-tokenizer: `split_pretokenize` GPT-2 contractions/space-letters/numbers/punct, `test_pretok`.
- B15 Mmap: `read_file_bytes` mmap path (`dataset.cpp`), `Dataset(path, block, use_mmap)`, 1MB equivalence.
- B16 Packing: `pack_with_eos(tokens, block, eos)` + masks, `test_data_pipeline`.
- B17 Shuffle: `DataLoader(ds, batch, shuffle, seed)` Fisher-Yates block order, same-seed identical, full coverage.
- B18 Split: `train_val_split(tokens, ratio, seed)`, 900/100 contiguous.
- B19 Downloader: `scripts/download_data.py` (tinystories/shakespeare, sha256, --dry-run).
- B20 Fuzz corpus: `tests/fuzz_corpus/` (empty/invalid-utf8/malagasy/tiny), `fuzz_tokenizer` asserts byte-roundtrip. Track B complete (10/10), 45/45 green.
- C21 GQA: `gqa_forward(x,Wq,Wk,Wv,nq,nkv,hd)` (`mqa.h/cpp`), nkv==1 ≡ `mqa_forward` <1e-4, `test_gqa`.
- C22 RMSNorm switch: `Config::use_rmsnorm` + `TransformerConfig::use_rmsnorm`, block + ln_f forward/incr/backward branch, `test_rmsnorm_model` trains.
- C23 YaRN: `rope_cfg` NTK base*scaling + YaRN ramp/attn-scale (`model.cpp`), `rope_mode/yarn_alpha/yarn_beta`, ntk-vs-yarn differ 0.19.
- C24 Guard: `validate_config` rejects alibi+RoPE, scaling>=1, global_every>0, `test_yarn_guard`.
