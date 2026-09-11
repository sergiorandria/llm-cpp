# Production-Ready Roadmap — 100 Atomic Tasks (= 100 Commits)

> Status 2026-09-11: `ctest` 31/31 green, per-layer KV-cache O(n) (`model.cpp:156`,
> `attention.cpp:100`), Flash tiled incremental (`flash_attention.cpp:44`,
> `FLASH_BLOCK_SIZE=32`), GGUF v3 bit-exact (`gguf.cpp:1`), speculative batched
> verify (`speculative.cpp:5`), beam logprob (`beam.cpp:1`), fused layernorm+gelu
> (`fused.cpp:1`), prefetch thread (`prefetch.cpp:1`), GPTQ scale_out (`gptq.cpp:8`),
> RLHF head+PPO (`rlhf.cpp:10,85`), staged pipeline+Adam (`pipeline.cpp:1`).
> This doc closes the gap to **production-ready**: correct, fast, servable,
> observable, releasable.

**Production definition:** deterministic training → perplexity drops on
TinyStories/Shakespeare; O(n) streaming inference; OpenAI-compatible server;
quantized GGUF/SafeTensors interop; ASAN/UBSAN + fuzz clean; Docker + CI matrix;
p95 latency + tokens/sec dashboard with regression gates.

**Commit rule:** 1 task = 1 commit = code + test + `docs/CHANGELOG.md` line.
Run `cmake -B build && cmake --build build -j && ctest --test-dir build`
per commit; buffer-touching commits also run `ENABLE_SANITIZERS=ON`.

---

## A. Correctness & Autograd hardening (A01–A10)

- [x] **A01 RMSNorm forward+backward** — `include/llm/layernorm.h`, `src/layernorm.cpp`.
  Add `rmsnorm(x, weight, eps)` + `rmsnorm_backward`. Test `test_rmsnorm.cpp`:
  finite-diff grad err <1e-3, equivalence to LayerNorm-with-zero-mean.
- [x] **A02 Central grad-check harness** — `tests/grad_check.h` (new).
  Helper `finite_diff(model, tokens, eps=1e-3)` reused by train/rlhf/pipeline tests.
  Commit refactors `test_train_step.cpp` + `test_grad_clip.cpp` to use it, no new logic.
- [x] **A03 Cross-entropy backward + label smoothing** — `src/loss.cpp`, `include/llm/loss.h`.
  Add `cross_entropy_backward(logits, target, smoothing=0.0/0.1)` returning dlogits.
  Test: smoothing=0 matches softmax-cross-entropy analytic; smoothing=0.1 sums to 0.
- [x] **A04 AdamW bias-correction + decoupled decay** — `src/optimizer.cpp`.
  Verify `m_hat/(sqrt(v_hat)+eps)` + decay only on weights (not biases/norms via name filter).
  Test `test_adamw.cpp`: 10 steps on Rosenbrock decreases loss, norms excluded.
- [x] **A05 Global-norm grad clipping** — `src/trainer.cpp` (+ `optimizer.h` helper `clip_grad_norm_`).
  Already asserted in `test_grad_clip.cpp`; extract to shared `clip_by_global_norm(params, 1.0)`.
  Test: norm>1 scaled to 1, norm<1 untouched.
- [x] **A06 Dropout determinism** — `src/tensor.cpp:dropout`.
  Thread `std::mt19937& rng` everywhere (attention, FFN); add `set_global_seed(seed)`.
  Test: same seed → identical forward; different seed → differs w.p. high.
- [x] **A07 NaN/Inf detector in train_step** — `src/trainer.cpp`.
  After `backward()`, scan `grad` for non-finite; on hit skip optimizer step + log + counter.
  Test: inject NaN grad, assert step skipped and counter++.
- [x] **A08 Deterministic mode flag** — `include/llm/model.h:Config{deterministic=false}`,
  `src/model.cpp`, `CMakeLists.txt` (`-DDETERMINISTIC` disables OpenMP nondeterminism).
  Test: two runs `TinyStories 50 steps` bit-identical when on.
- [x] **A09 Loss-scale guard for fp16 path** — `src/trainer.cpp`.
  Add `TrainConfig{loss_scale=1.0}`; scale loss before backward, unscale grads, skip on overflow.
  Test: scale=1024 finite path equals scale=1 within 1e-4.
- [x] **A10 Attention numerical parity** — `src/attention.cpp:100` vs `flash_attention.cpp:44`.
  Test `test_attn_parity.cpp`: full vs incremental vs flash-incremental logits diff <1e-4
  for K_len in {1, 33, 129, 512} (covers `K_len>128` dispatch boundary).

## B. Tokenizer & Data pipeline (B11–B20)

- [x] **B11 UTF-8 correctness** — `src/tokenizer.cpp`.
  `encode` must not split multi-byte codepoints at char level; `decode` must emit valid UTF-8.
  Test: Malagasy corpus (`data/malagasy/processed/input.txt`) roundtrip byte-exact.
- [x] **B12 HF `vocab.json`+`merges.txt` compat** — `src/tokenizer.cpp`, `include/llm/tokenizer.h`.
  Add `load_hf(vocab_path, merges_path)` + `save_hf(dir)`. Test: roundtrip our BPE equals HF ordering.
- [x] **B13 Streaming decode** — `include/llm/tokenizer.h:decode_incremental(ids, prev_bytes)`.
  Buffer incomplete UTF-8 tail across `generate_streaming` callbacks. Test: emoji split across 2 tokens.
- [x] **B14 GPT-2 pre-tokenizer regex** — `src/tokenizer.cpp`.
  Add `split_pretokenize(text)` (`'s|'t|'re| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`).
  Test `test_bpe_correctness.cpp` extension: `"Hello, world!"` splits as GPT-2.
- [x] **B15 Memmap dataset for large corpora** — `src/dataset.cpp`.
  `Dataset(path, block_size, mmap=true)` pages `data/*.bin` without full RAM load.
  Test: 100MB synthetic file loads with RSS <50MB (check `/proc/self/statm` in test).
- [x] **B16 Packing + EOS + masks** — `src/dataset.cpp`, `src/trainer.cpp`.
  Pack short docs with `<|endoftext|>` to `block_size`, emit `attention_mask`.
  Test: packed batch loss equals concatenated per-doc loss.
- [x] **B17 Multi-thread DataLoader** — `src/dataset.cpp` (+ reuse `PrefetchLoader` pattern).
  `DataLoader{num_workers=2}` background tokenization. Test: shuffled epoch covers all tokens once.
- [x] **B18 Train/val split + eval hook** — `src/dataset.cpp:train_val_split(0.99)`, `src/trainer.cpp`.
  `Trainer::evaluate(val_ds)` returns ppl. Test: overfit tiny set → train ppl < val ppl initially.
- [x] **B19 Data downloader + checksum** — `scripts/download_data.py` (new), `data/README.md`.
  Fetch TinyStories/Shakespeare with SHA256 verify. Test (CTest `download_smoke`): `--dry-run` passes offline.
- [x] **B20 Tokenizer fuzz corpus** — `tests/fuzz_tokenizer.cpp`, `.github/workflows/fuzz.yml`.
  Add seed corpus `tests/fuzz_corpus/*` (empty, long, invalid UTF-8, Malagasy lines).
  Gate: 60s libFuzzer run ASAN-clean.

## C. Architecture parity (C21–C30)

- [x] **C21 GQA generalize MQA** — `include/llm/transformer.h:Config{n_kv_heads}`,
  `src/attention.cpp`, `src/mqa.cpp` (keep `mqa_forward` as `n_kv_heads=1` alias).
  Test: `n_kv_heads=n_heads` ≡ MHA; `=1` ≡ MQA; `=4` KV-cache 3× smaller.
- [x] **C22 RMSNorm config switch** — `include/llm/model.h:Config{norm=RMS|Layer}`.
  Wire through `TransformerBlock` + `ln_f`. Test: RMS model trains 10 steps, ppl finite.
- [x] **C23 YaRN rope scaling** — `src/model.cpp:36` extend `rope_scaling` to `{ntk, yarn}` modes.
  Add `Config{rope_yarn_alpha, rope_yarn_beta}`. Test: 8k context (`rope_theta=50000`) forward no NaN.
- [x] **C24 ALiBi⊕RoPE guard** — `src/model.cpp`, `src/alibi.cpp`.
  `validate_config` rejects `alibi && pos_encoding==RoPE` (mutually exclusive). Test asserts exit code.
- [x] **C25 Hybrid sliding+global** — `src/sliding.cpp`, `include/llm/model.h:Config{sliding_window=512, global_every=4}`.
  Every 4th layer full attention. Test: 2k-token forward matches full-attn ppl within 5%.
- [x] **C26 MoE balance loss** — `src/moe.cpp`, `src/trainer.cpp`.
  Add `aux_loss = load_balance_coef * CV(gate)^2`, wire into `train_step`.
  Test: uniform routing → aux≈0; collapsed routing → aux>0.
- [x] **C27 LoRA merge/unmerge** — `src/lora.cpp` (exists, complete it).
  `apply_lora(model, r=8, alpha=16)` + `merge_lora()` for inference + `unmerge` for resume.
  Test: merged ≡ unmerged logits <1e-4; only adapter params receive grad.
- [x] **C28 True weight tying** — `src/model.cpp:tie_weights`.
  Replace copy-on-tie with shared storage (`shared_ptr<vector<float>>` or alias indices).
  Test: `lm_head` mutation visible in `wte` and vice versa; `num_parameters()` counts once.
- [x] **C29 Activation checkpointing** — `src/transformer.cpp`, `include/llm/model.h:Config{grad_checkpoint=true}`.
  Recompute FFN activations in backward to halve memory. Test: on/off grads match <1e-4.
- [x] **C30 Dynamic block_size extension** — `src/model.cpp:wpe_` resize + `load_config` interpolation.
  Load 1k checkpoint into 2k config via sinusoidal interp. Test: extended model forward finite.

## D. Training production (D31–D40)

- [x] **D31 Resume from checkpoint** — `src/checkpoint.cpp`, `src/optimizer.cpp`, `src/scheduler.cpp`.
  Persist `optimizer_state + scheduler_step + rng_state` in binary v4 header.
  Test: train 5 steps, resume 5 steps ≡ train 10 steps bit-exact (deterministic mode).
- [x] **D32 SafeTensors I/O** — `src/checkpoint.cpp` (+ `safetensors.cpp` new if >200 LOC).
  `save_safetensors`/`load_safetensors` (JSON header + little-endian f32).
  Test: safetensors→load→logits match binary v3 <1e-6.
- [x] **D33 HF `config.json` compat** — `src/config.cpp`.
  `load_hf_config(dir)` maps `n_layer/n_head/n_embd` → our `Config`. Test on GPT-2 124M shape.
- [x] **D34 Scheduler warmup+cosine+restarts** — `src/scheduler.cpp`.
  `CosineScheduler{warmup_steps, total_steps, min_lr, restarts}`. Test: lr curve matches formula.
- [x] **D35 Gradient accumulation** — `src/trainer.cpp:TrainConfig{grad_accum_steps=4}`.
  Accumulate `grad` over K micro-batches before `optimizer.step()`.
  Test: accum=4 × batch=1 ≡ batch=4 within 1e-4.
- [x] **D36 Single-node data-parallel** — `src/trainer.cpp` (threads, not MPI).
  Shard batch across `num_threads`, allreduce grads by mean. Test: 2-thread ≡ 1-thread <1e-4.
- [x] **D37 JSONL metrics logger** — `src/logging.cpp` (exists, extend).
  `train.log.jsonl` per-step `{step, loss, ppl, lr, grad_norm, tokens_sec}`.
  Test: 3-step run emits 3 parseable lines.
- [x] **D38 Best-checkpoint keeper** — `src/trainer.cpp:TrainConfig{keep_best_n=3, eval_every=100}`.
  Keep `best_ppl.bin` + rolling top-3. Test: simulated ppl sequence keeps correct files.
- [x] **D39 Eval-during-train** — `src/main.cpp:train` add `--eval_every --eval_data`.
  Prints `val_ppl` without breaking training RNG. Test: CLI smoke `--max_iters 2 --eval_every 1`.
- [x] **D40 Train CLI real flags** — `src/main.cpp:train` (today `max_iters=10` hardcoded).
  Wire `--max_iters --batch_size --lr --block_size --out`. Test: `--max_iters 1` finishes <30s tiny config.

## E. Inference engine (E41–E50)

- [x] **E41 Batched generate** — `include/llm/model.h:generate_batch(prompts, ...)` + padding mask.
  Test: batch of 4 ≡ 4× single greedy; ragged lengths handled.
- [x] **E42 Paged KV-block allocator** — `src/kv_cache.cpp:KVCacheConfig{paged, block_size=16}`.
  Replace contiguous `[n_embd, max_seq_len]` growth with block table (SoA path kept).
  Test: paged ≡ contiguous logits <1e-5; fragmentation test with interleaved sequences.
- [x] **E43 Real streaming tokens** — `src/model.cpp:generate_streaming`.
  Invoke `cb(token_id)` per decoded token (not at end). Test: callback sequence == returned vector.
- [x] **E44 Stop sequences + EOS** — `src/sampling.cpp:StopCriteria{eos_id, stop_ids, max_tokens}`.
  Test: generation halts at first stop string; EOS counted once.
- [x] **E45 Logprobs API** — `include/llm/model.h:generate_with_logprobs(...) → {tokens, logprobs, topk}`.
  Needed by beam (already does manual `log_softmax`) — expose cleanly. Test: probs sum to 1.
- [x] **E46 Frequency/presence penalties** — `src/sampling.cpp` extend `rep_penalty` to
  `{freq_penalty, pres_penalty}` (OpenAI semantics). Test: penalized token logit lowered exactly.
- [x] **E47 Logit bias / constrained vocab** — `src/sampling.cpp:logit_bias: map<int,float>`.
  Test: bias=-inf token never sampled over 100 draws temp=1.
- [x] **E48 Speculative speedup gate** — `tests/test_speculative.cpp` + `docs/BENCHMARK_DASHBOARD.md`.
  Record draft/target sizes + measured speedup; CI warns if batched verify slower than greedy.
- [x] **E49 Beam length penalty + early stop** — `src/beam.cpp:BeamConfig{len_penalty=0.6, early_stop=true}`.
  Score `logprob / len^penalty`. Test: penalty=0 prefers longer; large penalty prefers shorter.
- [x] **E50 Quantized GEMM inference** — `src/quantize.cpp` (fix `quantize.h:8` float-stub).
  Real `int8` storage + `int8_gemm(u8/s8, scale)` dispatch in `Tensor::matmul` when `model.is_quantized()`.
  Test: int8 ppl within 2% of fp32 on tiny model.

## F. Quant & compression (F51–F60)

- [x] **F51 Int8 Tensor dtype** — `include/llm/tensor.h:enum class DType{F32,I8}` + `Tensor{dtype,int8_data,scale}`.
  Migrate `quantize_model()` off float-shadow. Test: `dtype==I8` tensors reject fp32-only ops with clear error.
- [x] **F52 GPTQ scale persistence** — `src/gptq.cpp`, `src/checkpoint.cpp` binary v4.
  Save per-group `scale_out` alongside qlevels. Test: save→load→dequant error still <0.5.
- [x] **F53 AWQ scaling** — `src/gptq.cpp` (new `awq_scale(model, calib_data)`).
  Activation-aware per-channel scale before quant. Test: AWQ ppl ≤ naive int8 ppl on calib set.
- [x] **F54 GGUF Q4_K / Q8_0 types** — `src/gguf.cpp` extend beyond f32. (Done: Q8_0 + Q4_0 block-32; Q4_K super-blocks deferred — needs 256-elem scales/mins.)
  Write `GGML_TYPE_Q4_K` blocks + dequant on load. Test: Q8_0 roundtrip err <0.05.
- [x] **F55 Sparse matmul skip-zeros** — `src/prune.cpp` + `src/tensor.cpp:matmul_sparse(mask)` CSR path.
  Dispatch when `sparsity>0.5`. Test: 50% pruned model logits match dense <1e-4, faster on 512².
- [x] **F56 Teacher-logit cache for distill** — `src/distill.cpp`.
  `cache_teacher_logits(teacher, dataset, path)` to disk once. Test: cached ≡ live KL <1e-6.
- [x] **F57 Structured 2:4 pruning** — `src/prune.cpp:prune_2to4(model)`.
  Every 4 weights keep top-2 by magnitude + mask tensor. Test: mask satisfies 2:4 + ppl measured.
- [x] **F58 Quant calibration set** — `data/calib/` 1k-line bundled sample + `scripts/calibrate.py`.
  `quantize --calib` derives scales from activations, not random. Test: calib scales differ from default.
- [x] **F59 PPL-degradation CI gate** — `.github/workflows/quant.yml` (new).
  Fail if `int8 ppl - fp32 ppl > 5%` or `4bit err > 0.5` on tiny eval. Documents `<0.3 PPL loss` claim honestly.
- [x] **F60 Quantize CLI** — `src/main.cpp` `--bits {4,8} --group 128 --out quantized.bin`.
  Verify flag: dequant check printed. Test: CLI smoke produces loadable file.

## G. Serving & API (G61–G70)

- [x] **G61 OpenAI-compatible server** — `src/server.cpp` (new, cpp-httplib or Beast, vendored or FetchContent).
  `POST /v1/completions {prompt, max_tokens, temperature, top_p, stream}`.
  Test: python `requests` smoke vs `generate()` output match (temp=0).
- [x] **G62 SSE streaming** — `src/server.cpp:GET/POST stream=true`.
  `data: {token}` chunks + `data: [DONE]`. Test: streamed concat == non-stream body.
- [x] **G63 Python bindings** — `python/llm_cpp/__init__.py` + `python/bindings.cpp` (pybind11).
  `Tokenizer.encode/decode`, `GPT.forward/generate`. Test: `pytest python/tests/test_bind.py` green.
- [x] **G64 Production Dockerfile** — `Dockerfile` multi-stage (build → `ubuntu:24.04` runtime, non-root user).
  Image <500MB, `./llm-cpp --help` works. CI builds + pushes on tag.
- [x] **G65 Compose + healthcheck** — `docker-compose.yml`, `GET /healthz → {status, version, params}`.
  Test: `docker compose up -d && curl /healthz` in CI.
- [x] **G66 Generate CLI flags** — `src/main.cpp:generate --stream --stop --logprobs --seed --json`.
  `--json` emits `{text, tokens, logprobs, latency_ms}`. Test each flag smoke.
- [x] **G67 Config JSON schema** — `config/schema.json` + `validate_config` error strings
  (`"n_heads must divide n_embd"` not just `invalid config`). Test: 5 bad configs → 5 messages.
- [x] **G68 Model card generator** — `scripts/model_card.py --checkpoint → MODEL_CARD.md`
  (params, arch, ppl, quant, license). Test: golden file on tiny checkpoint.
- [x] **G69 Client examples** — `examples/{curl.sh, python_openai.py, node_fetch.mjs}`.
  CI runs curl example against local server.
- [x] **G70 Concurrency limit + shutdown** — `src/server.cpp:ServerConfig{max_concurrency=8}`.
  429 on overflow; SIGTERM drains. Test: 16 parallel reqs → 8×200 + 8×429.

## H. Observability & robustness (H71–H80)

- [x] **H71 Structured logging** — `src/logging.cpp:Logger{level=json}`.
  `{ts, level, msg, step, tokens_sec}` to stderr. Test: `LOG_INFO` line parses as JSON.
- [x] **H72 Prometheus metrics** — `src/server.cpp:GET /metrics`.
  `tokens_total, requests_total, latency_histogram, kv_cache_bytes`. Test: scrape after 3 gens.
- [x] **H73 Profiling timers** — `include/llm/profiling.h` (exists, wire it).
  Scoped `PROFILE("attn")` in `attention.cpp`, `transformer.cpp`, `model.cpp`.
  `bench` prints per-op ms. Test: `test_profiling` asserts all timers >0.
- [x] **H74 OOM / length guards** — `src/model.cpp:generate` early `prompt.size()+max_new > block_size → 400/Err`.
  Test: oversize prompt returns error, no alloc.
- [x] **H75 Input validation** — `src/server.cpp` + `src/sampling.cpp`.
  Clamp `temperature>=0, 0<top_p<=1, top_k>=0`; reject with message. Test table of 8 bad inputs.
- [x] **H76 Sanitizer CI** — `.github/workflows/san.yml` (`ENABLE_SANITIZERS=ON`, `ctest build_san`).
  Covers GGUF + tokenizer + speculative (already green locally per CHANGELOG 0.3.1–0.3.2).
- [x] **H77 GGUF fuzz harness** — `tests/fuzz_gguf.cpp` (magic/truncation/corrupt-shape mutations).
  ASAN-clean on 10k mutations; bad magic rejected, no crash (matches `test_gguf_roundtrip` bad-magic path).
- [x] **H78 Atomic checkpoint write** — `src/checkpoint.cpp:save_atomic` (write `.tmp` + `fsync` + `rename`).
  Test: kill -9 mid-write simulation leaves old checkpoint intact.
- [x] **H79 NaN-rollback** — `src/trainer.cpp:keep_last_good(params)` snapshot every `eval_every`.
  On NaN loss restore + halve LR. Test: injected NaN recovers and loss finite next step.
- [x] **H80 Generation audit log** — `src/server.cpp:ServerConfig{audit_log=path}` appends
  `{ts, prompt_hash, tokens, params}` (never raw PII by default). Test: 2 reqs → 2 lines, hash stable.

## I. Hardware & perf (I81–I90)

- [x] **I81 Thread-pool tuning** — `CMakeLists.txt` + `src/tensor.cpp`.
  `OMP_NUM_THREADS` + `LLM_THREADS` env, pin via `proc_bind`. Document in `docs/BUILD.md`.
  Test: `bench_matmul` scales ≥1.5× 1→4 threads on 512².
- [x] **I82 SIMD layernorm/softmax/gelu** — `src/tensor.cpp` via `USE_NUMPY_CPP` SIMD paths
  (verify AVX2/NEON dispatch, no scalar fallback on x86_64). Test: SIMD ≡ scalar <1e-6.
- [x] **I83 GEMM autotune extension** — `scripts/autotune_flash.py` → also tune `GEMM_BLOCK_M/N/K`.
  Emit `include/llm/gemm_config.h`. `docs/BENCHMARK_DASHBOARD.md` records before/after.
- [ ] **I84 FlashAttention-2 full-prefill** — `src/flash_attention.cpp:flash_attention_full(Q,K,V,causal)`.
  Tiled online-softmax over `seq×seq` (today only incremental `1×K` path). Test: ≡ naive attn <1e-4.
- [ ] **I85 KV-cache block reuse** — `src/kv_cache.cpp` free-list for evicted sequences (pairs with E42).
  Test: 100 sequential generations RSS stable (no growth).
- [ ] **I86 Tensor memory pool** — `src/tensor.cpp:TensorPool{acquire(shape), release}` for activations.
  `forward_with_hidden` reuses buffers. Test: 100 forwards no new `malloc` (count via hook).
- [ ] **I87 CUDA backend abstraction** — `include/llm/backend.h` (new) + `cmake/FindCUDA.cmake` + `-DUSE_CUDA=ON`.
  `Tensor::matmul` dispatches CPU/CUDA; CPU-only build unchanged. Test: skipped gracefully when no GPU.
- [ ] **I88 Quantized BLAS dispatch** — `src/tensor.cpp:100` extend `USE_OPENBLAS` path to `cblas_gemm_s8u8s32`
  when `F51` int8 present; fallback blocked int8 GEMM. Test: int8 GEMM ≡ fp32 within quant bound.
- [ ] **I89 Infer benchmark + gate** — `scripts/bench_infer.cpp` (new) `tokens/sec, ms/token, p50/p95`.
  `.github/workflows/bench.yml` fails on >10% regression vs `docs/BENCHMARK_DASHBOARD.md` baseline.
- [ ] **I90 Alignment + padding** — `src/tensor.cpp` 64B-aligned alloc, pad `vocab`/`n_embd` to mult of 32.
  Test: all `data.data()` 64B-aligned; padded matmul ≡ unpadded.

## J. Release & ecosystem (J91–J100)

- [ ] **J91 Version automation** — `include/llm/version.h` from `git describe` via CMake.
  `llm-cpp --version` prints `0.x.y+sha`. Test: version string matches `CHANGELOG` head.
- [ ] **J92 CHANGELOG gate** — `.github/workflows/lint.yml`: PR fails if no `docs/CHANGELOG.md` diff
  (except `docs/`-only PRs). Retroactively add Unreleased section for A–I as landed.
- [ ] **J93 clang-format + clang-tidy** — `.clang-format` exists; add `.clang-tidy` + pre-commit hook.
  `scripts/lint.sh` runs both. Test: CI lint green (fix `build/` warnings first).
- [ ] **J94 Coverage gate** — `cmake -DENABLE_COVERAGE=ON` + `lcov`; fail if line coverage <80%
  (target 90% by 1.0). Upload to Codecov. Exclude `examples/`, `build*/`.
- [ ] **J95 Overfit integration test** — `tests/test_overfit.cpp` (new, CTest `overfit`, timeout 300s).
  Tiny Shakespeare 200 lines, 2-layer 64-embd, 50 steps → `ppl` strictly decreases, final < initial/1.5.
- [ ] **J96 Quantized GGUF roundtrip** — extend `tests/test_gguf_roundtrip.cpp` to Q8_0 path (pairs with F54).
  Assert dequant err bound + greedy decode identical to fp32 on fixed prompt.
- [ ] **J97 SBOM + license scan** — `scripts/sbom.sh` (Syft or `pip-licenses` for numpy-cpp pin).
  `THIRD_PARTY.md` lists MIT/Apache deps. CI fails on GPL-incompatible addition.
- [ ] **J98 Build matrix** — `.github/workflows/ci.yml` extend to `{gcc-11,gcc-13,clang-15} × {Debug,Release} × {ON,OFF: USE_NUMPY_CPP, USE_OPENBLAS}`.
  All green + `ctest` 31+ new tests.
- [ ] **J99 Release artifacts** — `.github/workflows/release.yml` on tag `v*`:
  `llm-cpp-linux-x64.tar.gz` (binary + `include/` + `config/schema.json` + README).
  Test: download artifact, `--help` exits 0.
- [ ] **J100 1.0 sign-off doc** — `docs/PRODUCTION.md` checklist (all 99 above ticked, bench numbers,
  known limits: CPU-only unless I87 landed, single-node, EN + Malagasy eval).
  Maintainer signs + tags `v1.0.0`.

---

## Suggested commit order (dependency-safe)

1. A01–A10 (no API breaks except C28 — do C28 early if wanted, it touches param counting).
2. B11–B20 (tokenizer/data, independent).
3. C21–C30, then D31–D40 (C28 → D31/D32 order matters: tie before persist).
4. E41–E50 + F51–F60 interleaved (F51 before E50/F54/F60).
5. G61–G70 after E41–E47 stable.
6. H71–H80 anytime; H76/H77 before G61 (fuzz server inputs).
7. I81–I90 after E/H benchmarks exist (I89 needs dashboard).
8. J91–J100 last (gates assume features present).

Estimated: ~100 commits, ~2–5 tests each, `ctest` stays green throughout.
