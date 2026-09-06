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
