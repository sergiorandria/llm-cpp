# PLAN 2 — Training-Path Stub Elimination

**Owner: Muse Spark 1.2 (track 2)**
**Scope:** `src/gptq.cpp`, `src/rlhf.cpp`, `src/pipeline.cpp`,
`src/eval.cpp` + their headers in `include/llm/` + `tests/test_gptq.cpp`,
`tests/test_rlhf.cpp`, `tests/test_pipeline.cpp`, `tests/test_eval.cpp` +
`README.md`'s checklist lines for these specific features only.

**Out of scope:** `src/gguf.cpp`, `src/speculative.cpp`, `src/beam.cpp`,
`src/fused.cpp`, `src/prefetch.cpp`, `src/model.cpp` core forward/backward,
`src/optimizer.cpp` (may be *called*, not modified — see task 2/3). Track 1
owns the inference-path files.

## Why this track exists
Same pattern as Track 1, on the training/compression side — header comments
promise more than the `.cpp` delivers:

- `include/llm/gptq.h` claims "4× memory, <0.3 PPL loss at 7B", but
  `quantize_4bit()` computes a per-group `scale` internally and then
  **discards it** — the function returns only the quantized levels, with no
  way for a caller to recover the scale needed by `dequantize_4bit()`. The
  round trip is only possible today if a caller re-derives the exact same
  scale independently, which is fragile and likely why no real caller uses
  this outside the test.
- `include/llm/rlhf.h` is honest that PPO is "(stub)" in the header, but the
  gap is worth closing: `RewardModel` never trains a reward head at all — the
  constructor comment says it would build one from `n_embd` but doesn't, and
  `score()` just runs the frozen base model and returns average log-prob of
  the completion, which is a language-model-likelihood proxy, not a learned
  preference reward. `ppo_step` has no policy ratio, no clipping, no
  reference-policy KL term (despite `RewardConfig::kl_coef` existing and
  being unused), and no advantage baseline — it's REINFORCE with a raw
  clamped reward, not PPO.
- `include/llm/pipeline.h` claims "split n_layers across stages ... overlap
  comm with compute" but `Pipeline::train_step` never splits the model by
  layer — `stages_` is read once via `(void)stages_` and never otherwise
  used. What's implemented is plain gradient accumulation over
  micro-batches, sequential, on one "device". No pipelining exists.
- `eval.cpp`'s `evaluate()` hardcodes `mmlu`/`hellaswag` to `0` — perplexity
  is real, but the function signature (`mmlu_data`) and `EvalResult` fields
  imply benchmark support that doesn't exist.
- Both `ppo_step` and `Pipeline::train_step` hand-roll their own inline SGD
  update loop instead of using the project's own `src/optimizer.cpp`
  (Adam/AdamW), so RLHF and pipeline training silently skip whatever
  optimizer the rest of the codebase uses.


## Tasks

1. **Fix the GPTQ round trip.** Change `quantize_4bit`'s signature (or add
   an overload) to also output the per-group `scale` tensor it already
   computes internally, e.g. `Tensor quantize_4bit(const Tensor& x, Tensor&
   scale_out, size_t group=128)`. Update `test_gptq.cpp` to actually call
   `dequantize_4bit` with the *returned* scale (not a re-derived one) and
   assert reconstruction error is within a stated bound — this is the test
   that would have caught the missing scale output.

2. **Give RLHF a real reward head.** Add a small trainable linear layer
   (`n_embd -> 1`) to `RewardModel`, initialized from the base model's
   config (get `n_embd` from `GPT`'s public config accessor rather than
   guessing 768), and a `RewardModel::train_step` that fits it on
   preference pairs (chosen/rejected) via a pairwise logistic loss —
   the standard Bradley-Terry RLHF reward objective. Keep `score()` as the
   inference-time entry point once the head is trained.

3. **Turn `ppo_step` into actual PPO.** Add: an old/reference policy
   snapshot (frozen copy or logits cached before the update), the PPO
   clipped-ratio objective (`min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`),
   an advantage estimate (reward minus a running baseline, since there's no
   value head yet — a full value head is a reasonable follow-up, not
   required for this task), and wire in the existing `RewardConfig::kl_coef`
   as a real KL penalty against the reference policy's log-probs instead of
   leaving it declared-but-unused. Replace the inline hand-rolled SGD loop
   with a call into `src/optimizer.cpp`'s existing optimizer.

4. **Turn `Pipeline::train_step` into real pipelining, or rename it.**
   Either implement an actual layer-range split (`GPT`'s blocks partitioned
   across `stages_` groups, each stage's forward/backward computed and
   handed off to the next — even single-threaded, this should be a real
   staged computation, not accumulation) with overlap where feasible, or —
   if true pipeline parallelism is out of scope for a single-process build —
   rename the class/API to `GradientAccumulator` and drop the "pipeline
   parallel" framing from the header comment and README until real staging
   exists. Replace its inline SGD loop with the shared optimizer as in
   task 3.

5. **Real MMLU/HellaSwag eval, or honest scoping.** Either wire in a small
   bundled sample of MMLU/HellaSwag-format data (multiple-choice, scored by
   comparing log-likelihood of each choice) under `data/` with a real
   scoring loop in `evaluate()`, or rename `EvalResult::mmlu`/`hellaswag`
   and the `mmlu_data` parameter to make clear only perplexity is measured
   today, removing the always-0 fields until real benchmark support exists.

## Verification
`ctest --test-dir build` must stay green throughout. For tasks 2–4, add a
determinism/finite-difference gradient check in the same style
`test_train_step.cpp` already uses, since these paths do real backward
passes and silent gradient bugs are easy to introduce.
