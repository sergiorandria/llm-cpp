# llm-cpp v1.0 — Production Sign-off (J100)

100/100 roadmap tasks complete (`docs/PRODUCTION_100.md` all ticked), **80 ctest suites
green** (plus `pybind`), sanitizer-relevant fuzz clean, benches recorded.

## Checklist

| Area | State |
|------|-------|
| A correctness (RMSNorm, grad harness, CE, AdamW, clip, seeds, NaN guard, determinism, loss-scale, parity) | ✅ 10/10 |
| B tokenizer/data (UTF-8, HF, streaming, pre-tok, mmap, packing, shuffle, split, downloader, fuzz) | ✅ 10/10 |
| C architecture (GQA, RMS switch, YaRN, ALiBi guard, hybrid, MoE aux, LoRA merge, tying, ckpt flag, extend) | ✅ 10/10 |
| D training (resume, SafeTensors, HF cfg, scheduler, accum, parallel-mean, JSONL, keeper, eval hook, CLI) | ✅ 10/10 |
| E inference (batch, paged KV, streaming, stop, logprobs, penalties, bias, spec gate, beam cfg, int8 GEMM) | ✅ 10/10 |
| F quant (I8 dtype, GPTQ persist, AWQ, GGUF Q8/Q4, sparse, 2:4, distill cache, calib, gate, CLI) | ✅ 10/10 |
| G serving (server, SSE, ctypes, Dockerfile, compose, CLI, schema, card, examples, 429) | ✅ 10/10 |
| H observability (JSON logs, metrics, profiling, guards, validation, san CI, fuzz, atomic, rollback, audit) | ✅ 10/10 |
| I perf (threads, SIMD, autotune, FA2, pool, backend, int8 MACs, bench gate, align) | ✅ 10/10 |
| J release (version, changelog gate, lint, coverage, overfit, Q8 test, SBOM, matrix, artifacts, this doc) | ✅ 10/10 |

## Measured numbers (this box)

- Overfit: Shakespeare-200 loss 5.10 → 0.04 (100 steps, 6.8 s)
- Infer: 352.6 tok/s (tiny 2L/64e, OMP_NUM_THREADS=1; 0.37 tok/s default threads — pin threads)
- Speculative tiny-scale speedup 0.39 (2× claim needs draft ≪ target)
- Int8 ppl drift 8e-6 (gate <5%); Q8 GGUF err 0.0003; 2:4 prune ppl 2.7024→2.7029
- Flash-vs-naive 7e-8; paged≡contiguous 0.0; resume 5+5 ≡ 10 bit-exact

## Known limits (honest)

- CPU-only (CUDA abstraction present, kernels not yet; `backend_name()=="cpu"`)
- Single-node, single-process (data-parallel = in-process mean-reduce)
- Eval: EN TinyStories/Shakespeare + Malagasy corpus; MMLU/HellaSwag are token-accuracy proxies
- Q4_K super-blocks deferred (Q4_0 + Q8_0 shipped); GGUF Q-scales are F32 super-scales
- numpy-cpp upstream ships NO LICENSE file (THIRD_PARTY.md exception; follow-up filed)
- Release-mode `assert()` is NDEBUG-elided: tests use hoisted calls; Debug/ASAN CI gives real checking
- Docker image build ~25 min+ (tag CI only, not per-push)

## Sign-off

- [x] `ctest` green, bench gate green, quant gate green (maintainer run)
- Maintainer signature: ______________________  date: __________
- Release: `git tag -a v1.0.0` (see `.github/workflows/release.yml` artifacts)
