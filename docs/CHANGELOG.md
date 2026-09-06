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
