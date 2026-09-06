# PLAN 1 — Inference-Path Stub Elimination

**Owner: Muse Spark 1.2 (track 1)**
**Scope:** `src/gguf.cpp`, `src/speculative.cpp`, `src/beam.cpp`,
`src/fused.cpp`, `src/prefetch.cpp` + their headers in `include/llm/` +
`tests/test_*` for these (add where missing) + `README.md`'s checklist
lines for these specific features only.

**Out of scope:** `src/gptq.cpp`, `src/rlhf.cpp`, `src/pipeline.cpp`,
`src/eval.cpp`, `src/optimizer.cpp`, `src/trainer.cpp`, `src/model.cpp`
core forward/backward. Track 2 owns those.

## Why this track exists
`docs/CHANGELOG.md` itself admits the "100 tasks" milestone was partly
fabricated (11 headers with no `.cpp`, not compiled) and that 0.3.0 fixed
that — but several of the fixes are themselves stubs whose header comments
promise more than the code does:

- `include/llm/gguf.h` promises save/load; `load_gguf()` prints
  `"[gguf] load stub"` and returns `true` without touching `model` at all.
  `save_gguf()` only prints parameter count, never serializes a tensor.
  Any code path that "round-trips" a model through GGUF is currently lying.
- `include/llm/speculative.h` claims "verifies in parallel" and "2× speedup
  ... draft 10M vs target 124M"; `speculative.cpp` says outright in a
  comment `"here sequential for stub"` and calls `target_.generate()`
  once per accepted-token-run — this is not just unparallelized, it likely
  runs **slower** than plain target-only decoding, the opposite of the
  header's claim.
- `beam.cpp`'s `beam_search` never tracks or compares cumulative
  log-probability across beams and never prunes — it just runs
  `beam_width` independent stochastic rollouts and returns `beams[0]`
  unconditionally. This is not beam search.
- `fused.cpp`'s `fused_gelu` calls `x.gelu()` with a comment admitting
  "keep scalar for now but fused" — it fuses nothing.
- `prefetch.cpp`'s `PrefetchLoader` is fully synchronous (no thread, no
  lookahead buffer) despite the name and the `prefetch_` constructor
  parameter, which is stored but never used.


## Tasks

1. **Real GGUF I/O.** Implement actual tensor serialization in `save_gguf`
   (name, shape, dtype, raw data per `GPT::parameters()`) and a matching
   `load_gguf` that allocates/validates shapes and populates tensor data —
   not just a config check. Either implement the real GGUF binary format
   (header magic + KV metadata + tensor info + data, see llama.cpp's spec)
   or, if that's out of scope for now, rename the API to something honest
   (e.g. `save_checkpoint_v4`) and drop the GGUF claim until the real format
   is implemented. Add `test_gguf_roundtrip.cpp` asserting a saved-then-
   loaded model produces identical logits on a fixed prompt.

2. **Real speculative decoding.** Rewrite `SpeculativeDecoder::generate` to
   do a single target forward pass over the full draft-proposed sequence
   per round (batched verification), compare `argmax` (or sampled token
   under matching distribution) per position against the draft tokens, and
   accept the longest matching prefix — this is the actual algorithm the
   header describes. Add a benchmark or test asserting output token-for-
   token equivalence with greedy target-only decoding (temp=0), and a timing
   assertion or comment noting expected speedup only holds once this is done.

3. **Real beam search.** Track cumulative log-probability per beam
   (need per-step logits, not just sampled tokens — extend `GPT::generate`
   or add a lower-level "step" API returning logits if one doesn't exist),
   expand each beam's top-k continuations, keep the `beam_width` highest-
   scoring sequences each step, and return the best-scoring completed beam
   at the end instead of `beams[0]`. Add `test_beam_search.cpp` asserting
   beam search's chosen sequence has cumulative log-prob >= greedy decoding's.

4. **Real (or honestly-scoped) fusion.** Either implement `fused_gelu` as an
   actual fused op (e.g. fold into the preceding matmul's output loop to
   avoid a second full pass over the tensor, or a genuine SIMD tanh-approx
   path via `numpy-cpp`), or rename it to make clear it's currently an alias
   and remove the "fused" framing until real fusion lands.

5. **Real (or honestly-scoped) prefetching.** Either implement an actual
   background-thread lookahead buffer in `PrefetchLoader` (a small
   producer/consumer queue filling up to `prefetch_` batches ahead of
   consumption), or rename the class/drop the unused `prefetch_` parameter
   so the API doesn't promise concurrency it doesn't have.

6. **Update `README.md`** to reflect only what's true after 1–5 — no
   feature line should claim behavior (parallel verification, fusion,
   prefetching) the corresponding source doesn't implement.

## Verification
`ctest --test-dir build` must stay green throughout; add new tests
incrementally per task rather than at the end, and run
`ENABLE_SANITIZERS=ON` builds for anything touching raw buffers (GGUF
serialization especially).
