// tests/test_eval_mc.cpp — real MMLU/HellaSwag multiple-choice eval.
// 1. Pure scoring mechanics on hand-built logits (no model).
// 2. JSONL loaders on data/eval/*_sample.jsonl (incl. skip path + "B" answer).
// 3. End-to-end: train tiny tokenizer on sample texts, encode, score with a
//    tiny model — asserts wiring (range, determinism, truncation), not smarts.
// Explicit checks (Release NDEBUG elides assert bodies).
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "llm/eval.h"
#include "llm/tokenizer.h"

#define CHECK(cond, msg)                          \
    do {                                          \
        if (!(cond)) {                            \
            std::cerr << "FAIL: " << msg << "\n"; \
            return 1;                             \
        }                                         \
    } while (0)

int main() {
    using namespace llm;

    // 1. Mechanics on hand-built logits: rows [ctx][choice0].
    {
        Tensor logits({2, 3}, 0.0f);
        logits.data = {1.0f, 2.0f, 3.0f, 0.5f, 0.1f, -0.5f};
        float got = choice_loglik_from_logits(logits, 1, {0});
        double mx = 0.5, lse = mx + std::log(std::exp(0.0) + std::exp(-0.4) + std::exp(-1.0));
        float want = (float)(0.5 - lse);
        CHECK(std::abs(got - want) < 1e-5f, "choice_loglik math mismatch");
        // two-token choice sums both positions
        float got2 = choice_loglik_from_logits(logits, 0, {2, 0});
        double mx0 = 3.0, lse0 = mx0 + std::log(std::exp(-2.0) + std::exp(-1.0) + std::exp(0.0));
        float want2 = (float)((3.0 - lse0) + (0.5 - lse));
        CHECK(std::abs(got2 - want2) < 1e-5f, "two-token choice mismatch");
        CHECK(choice_loglik_from_logits(logits, 1, {}) < -1e29f, "empty choice not rejected");
        CHECK(choice_loglik_from_logits(logits, 2, {0}) < -1e29f, "overflow not rejected");
        CHECK(pick_best({0.1f, 0.5f, 0.3f}) == 1, "pick_best wrong");
        CHECK(pick_best({0.5f, 0.5f, 0.1f}) == 0, "pick_best tie not first");
    }

    // 2. Loaders on bundled samples.
    size_t skipped = 0;
    auto mmlu = load_mmlu_jsonl("data/eval/mmlu_sample.jsonl", &skipped);
    CHECK(mmlu.size() == 5, "mmlu sample count");
    CHECK(skipped == 0, "mmlu unexpected skips");
    CHECK(mmlu[3].answer == 1, "letter answer B not parsed to 1");
    CHECK(mmlu[0].choices.size() == 4, "mmlu choices count");
    auto hs = load_hellaswag_jsonl("data/eval/hellaswag_sample.jsonl", &skipped);
    CHECK(hs.size() == 7, "hellaswag sample count");
    CHECK(hs[6].answer == 0, "digit-string label not parsed");
    CHECK(skipped == 1, "hellaswag malformed line not skipped");
    CHECK(hs[5].context == "A woman pours flour into a bowl. Next she", "ctx_a/b join wrong");
    auto missing = load_mmlu_jsonl("data/eval/does_not_exist.jsonl", &skipped);
    CHECK(missing.empty() && skipped == 0, "missing file not empty");

    // 3. End-to-end wiring with a tiny model + trained tokenizer.
    {
        std::string corpus;
        for (auto& it : mmlu) {
            corpus += it.context + "\n";
            for (auto& c : it.choices) corpus += c + "\n";
        }
        for (auto& it : hs) {
            corpus += it.context + "\n";
            for (auto& c : it.choices) corpus += c + "\n";
        }
        Tokenizer tok(512);
        tok.train(corpus, 64);
        auto enc = [&](const std::string& s) {
            auto ids = tok.encode(s);
            for (auto& id : ids) id = ((id % 512) + 512) % 512;
            return ids;
        };
        std::vector<MCItem> m_items, h_items;
        for (auto& it : mmlu) {
            MCItem m;
            m.context = enc(it.context);
            for (auto& c : it.choices) m.choices.push_back(enc(c));
            m.answer = it.answer;
            m_items.push_back(std::move(m));
        }
        for (auto& it : hs) {
            MCItem m;
            m.context = enc(it.context);
            for (auto& c : it.choices) m.choices.push_back(enc(c));
            m.answer = it.answer;
            h_items.push_back(std::move(m));
        }
        Config cfg;
        cfg.vocab_size = 512;
        cfg.n_embd = 16;
        cfg.n_heads = 2;
        cfg.n_layers = 1;
        cfg.block_size = 64;
        GPT model(cfg);
        MCScore sm = score_mc_items(model, m_items);
        MCScore sh = score_mc_items(model, h_items);
        CHECK(sm.scored == 5 && sh.scored == 7, "not all items scored");
        CHECK(sm.accuracy >= 0 && sm.accuracy <= 1, "mmlu accuracy out of range");
        CHECK(sh.accuracy >= 0 && sh.accuracy <= 1, "hs accuracy out of range");
        MCScore sm2 = score_mc_items(model, m_items);
        CHECK(sm2.picks == sm.picks, "scoring not deterministic");
        std::cout << "mc wiring: mmlu acc=" << sm.accuracy << " hs acc=" << sh.accuracy << "\n";

        // evaluate_mc fills real MC accuracies over the legacy proxy.
        std::vector<std::vector<int>> seqs = {{1, 2, 3, 4}};
        EvalResult r = evaluate_mc(model, seqs, m_items, h_items);
        CHECK(r.mmlu == sm.accuracy && r.hellaswag == sh.accuracy, "evaluate_mc mismatch");
        CHECK(r.ppl > 0 && std::isfinite(r.ppl), "ppl invalid");
        // Empty MC sets -> legacy proxy path preserved.
        EvalResult r0 = evaluate_mc(model, seqs, {}, {});
        EvalResult rl = evaluate(model, seqs);
        CHECK(r0.mmlu == rl.mmlu && r0.hellaswag == rl.hellaswag, "legacy path changed");

        // Truncation: tiny block still scores (context shrinks, choice kept).
        Config tiny = cfg;
        tiny.block_size = 8;
        GPT small(tiny);
        MCItem it = m_items[0];
        float s = choice_loglik(small, it.context, it.choices[0]);
        CHECK(std::isfinite(s), "truncated scoring non-finite");
        // Choice longer than block -> rejected, never crashes.
        std::vector<int> big_choice(16, 1);
        CHECK(choice_loglik(small, {1}, big_choice) < -1e29f, "oversize choice not rejected");
    }

    std::cout << "test_eval_mc passed\n";
    return 0;
}
