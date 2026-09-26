# Eval samples — MMLU / HellaSwag (real multiple-choice)

Tiny handcrafted samples in the authentic upstream field shapes so the
loader and log-likelihood choice scoring (`llm/eval.h`) are exercised on
real files. A random-init model scores ~chance here — these samples test
*wiring*, not model quality.

## Formats

**MMLU** (`mmlu_sample.jsonl`, one object per line — HF `cais/mmlu` shape):
```json
{"question": "...", "choices": ["...", "...", "...", "..."], "answer": 1, "subject": "..."}
```
`answer` accepts an int index or a letter `"A"`–`"D"`. Extra keys ignored.

**HellaSwag** (`hellaswag_sample.jsonl` — HF `Rowan/hellaswag` shape):
```json
{"ctx": "...", "endings": ["...", "...", "...", "..."], "label": 0}
```
`ctx_a`/`ctx_b` pairs are joined with a space when `ctx` is absent.
The last line of the sample is intentionally malformed (bad label) to
exercise the loader's skip path (`n_skipped`).

## Scoring

`choice_loglik(model, ctx, choice)` = sum of choice-token log-probs from a
single forward pass over `ctx + choice` (choice never truncated; context
truncates left). Accuracy = fraction of items where the argmax-choice
matches the label (`pick_best`, first on ties).

## Using full datasets

Fetch them with `scripts/download_eval.py` (stdlib only, verified live):
```bash
python3 scripts/download_eval.py --dry-run          # no network
python3 scripts/download_eval.py --dataset mmlu --subjects astronomy,global_facts
python3 scripts/download_eval.py --dataset hellaswag   # validation split, ~10k rows
```
- MMLU upstream is parquet (no stdlib reader), so the script pages the HF
  datasets-server JSON API and converts rows to the shape above.
- HellaSwag validation comes from `Rowan/hellaswag` the same way; its
  `label` arrives as a digit-string (`"3"`) — normalized to int on write,
  and the loader accepts both forms.
- Fetches are all-or-nothing per dataset (retries + completeness check;
  rate-limit truncation never saves a partial file). Outputs land in
  `data/eval/full/` (gitignored — only these samples are tracked).

Then point the loaders at the files: `load_mmlu_jsonl(path)`,
`load_hellaswag_jsonl(path)`. Texts are encoded with your own tokenizer
(`Tokenizer::train` + `encode`); ids are clamped into the model vocab at
scoring time. Full upstream sets need an HF-capable tokenizer alignment
for comparable numbers — see `test_eval_mc.cpp` for the intended pipeline.
