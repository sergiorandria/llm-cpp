#pragma once
#include "model.h"
namespace llm {
// Evaluation harness: perplexity is real (exp(avg cross-entropy)).
// mmlu/hellaswag are multiple-choice accuracies when MC items are provided
// (real log-likelihood choice scoring, LM-eval style), else fall back to the
// legacy next-token-accuracy proxy over sequence data. Bundled samples under
// data/eval/ (authentic MMLU/HellaSwag JSONL field names) exercise the loader.
struct EvalResult { float mmlu=0, hellaswag=0, ppl=0; };
EvalResult evaluate(const GPT& model, const std::vector<std::vector<int>>& data);

// Multiple-choice item: pick the choice with highest log-likelihood given
// the context. MMLU: context = question stem, choices = answers.
// HellaSwag: context = ctx, choices = endings.
struct MCItem {
    std::vector<int> context;
    std::vector<std::vector<int>> choices;
    int answer = 0;  // index into choices
};
// Pure scoring on precomputed logits for (context + choice): sum of choice
// token log-probs at their positions. ctx_len = context length (choice
// occupies [ctx_len, ctx_len + choice.size())). Testable without a model.
float choice_loglik_from_logits(const Tensor& logits, size_t ctx_len,
                                const std::vector<int>& choice);
// Full MC scoring with one forward pass of (context + choice) per choice.
// Long inputs truncate LEFT (context shrinks, choice tail never cut).
float choice_loglik(const GPT& model, const std::vector<int>& context,
                    const std::vector<int>& choice);
int pick_best(const std::vector<float>& scores);  // argmax, first on ties
// Accuracy over items (items with no valid choices are skipped).
// Returns {accuracy, num_scored}; picks[i] = chosen index per scored item.
struct MCScore { float accuracy = 0; size_t scored = 0; std::vector<int> picks; };
MCScore score_mc_items(const GPT& model, const std::vector<MCItem>& items);
// Combined report: ppl (+ legacy proxy when no MC data) with real MC
// accuracies filled from the provided item sets.
EvalResult evaluate_mc(const GPT& model, const std::vector<std::vector<int>>& data,
                       const std::vector<MCItem>& mmlu, const std::vector<MCItem>& hellaswag);

// Minimal JSONL loaders (no dependency; strict subset sufficient for the
// documented formats — malformed lines skipped, count returned via n_skipped).
// MMLU: {"question": str, "choices": [str x N], "answer": int|"A"-"D", ...}
// HellaSwag: {"ctx": str, "endings": [str x N], "label": int, ...}
struct MCTextItem {
    std::string context;
    std::vector<std::string> choices;
    int answer = 0;
};
std::vector<MCTextItem> load_mmlu_jsonl(const std::string& path, size_t* n_skipped = nullptr);
std::vector<MCTextItem> load_hellaswag_jsonl(const std::string& path, size_t* n_skipped = nullptr);
}
