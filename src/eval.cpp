#include "llm/eval.h"

#include <cmath>
#include <fstream>
#include <sstream>

#include "llm/loss.h"

namespace llm {

float choice_loglik_from_logits(const Tensor& logits, size_t ctx_len,
                                const std::vector<int>& choice) {
    if (choice.empty() || logits.shape.size() != 2) return -1e30f;
    size_t T = logits.shape[0], V = logits.shape[1];
    if (V == 0 || ctx_len + choice.size() > T) return -1e30f;
    float total = 0;
    for (size_t k = 0; k < choice.size(); ++k) {
        size_t pos = ctx_len + k;
        // Log-softmax row `pos` (double accumulation for stability)
        double mx = logits.data[pos * V];
        for (size_t j = 1; j < V; ++j) mx = std::max(mx, (double)logits.data[pos * V + j]);
        double sum = 0;
        for (size_t j = 0; j < V; ++j) sum += std::exp((double)logits.data[pos * V + j] - mx);
        double lse = mx + std::log(sum);
        int tok = choice[k];
        int vi = (int)V;
        tok = ((tok % vi) + vi) % vi;  // clamp OOV like the rest of the codebase
        total += (float)((double)logits.data[pos * V + (size_t)tok] - lse);
    }
    return total;
}

float choice_loglik(const GPT& model, const std::vector<int>& context,
                    const std::vector<int>& choice) {
    if (choice.empty()) return -1e30f;
    const auto& cfg = model.config();
    std::vector<int> full = context;
    full.insert(full.end(), choice.begin(), choice.end());
    size_t ctx_len = context.size();
    if (full.size() > cfg.block_size) {
        // Truncate LEFT: keep the choice tail intact, shrink context.
        size_t drop = full.size() - cfg.block_size;
        if (drop >= ctx_len) return -1e30f;  // choice alone doesn't fit
        full.erase(full.begin(), full.begin() + (long)drop);
        ctx_len -= drop;
    }
    if (full.empty()) return -1e30f;
    Tensor logits = model.forward(full);
    return choice_loglik_from_logits(logits, ctx_len, choice);
}

int pick_best(const std::vector<float>& scores) {
    int best = 0;
    for (size_t i = 1; i < scores.size(); ++i)
        if (scores[i] > scores[best]) best = (int)i;
    return best;
}

MCScore score_mc_items(const GPT& model, const std::vector<MCItem>& items) {
    MCScore out;
    size_t correct = 0;
    for (auto& it : items) {
        std::vector<float> scores;
        std::vector<int> valid;
        for (size_t c = 0; c < it.choices.size(); ++c) {
            if (it.choices[c].empty()) continue;
            scores.push_back(choice_loglik(model, it.context, it.choices[c]));
            valid.push_back((int)c);
        }
        if (valid.empty()) continue;  // no scorable choices
        int pick_valid = pick_best(scores);
        int pick = valid[(size_t)pick_valid];
        out.picks.push_back(pick);
        if (pick == it.answer) ++correct;
        ++out.scored;
    }
    if (out.scored > 0) out.accuracy = (float)correct / (float)out.scored;
    return out;
}

EvalResult evaluate_mc(const GPT& model, const std::vector<std::vector<int>>& data,
                       const std::vector<MCItem>& mmlu, const std::vector<MCItem>& hellaswag) {
    EvalResult res = evaluate(model, data);
    if (!mmlu.empty()) res.mmlu = score_mc_items(model, mmlu).accuracy;
    if (!hellaswag.empty()) res.hellaswag = score_mc_items(model, hellaswag).accuracy;
    return res;
}

// ── Minimal JSONL field extractors ──
static bool extract_string(const std::string& line, const std::string& key, std::string& val) {
    std::string pat = "\"" + key + "\"";
    size_t p = line.find(pat);
    if (p == std::string::npos) return false;
    p = line.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p = line.find('"', p + 1);
    if (p == std::string::npos) return false;
    std::string o;
    for (size_t i = p + 1; i < line.size(); ++i) {
        char c = line[i];
        if (c == '\\' && i + 1 < line.size()) {
            char e = line[++i];
            o += (e == 'n') ? '\n' : (e == 't') ? '\t' : e;  // \" \\ etc. pass through
        } else if (c == '"') {
            val = o;
            return true;
        } else {
            o += c;
        }
    }
    return false;
}

static bool extract_string_array(const std::string& line, const std::string& key,
                                 std::vector<std::string>& val) {
    std::string pat = "\"" + key + "\"";
    size_t p = line.find(pat);
    if (p == std::string::npos) return false;
    p = line.find('[', p + pat.size());
    if (p == std::string::npos) return false;
    std::vector<std::string> o;
    for (size_t i = p + 1; i < line.size();) {
        while (i < line.size() && (line[i] == ' ' || line[i] == ',')) ++i;
        if (i >= line.size()) return false;
        if (line[i] == ']') {
            val = o;
            return true;
        }
        if (line[i] != '"') return false;
        std::string s;
        bool closed = false;
        for (size_t j = i + 1; j < line.size(); ++j) {
            char c = line[j];
            if (c == '\\' && j + 1 < line.size()) {
                char e = line[++j];
                s += (e == 'n') ? '\n' : (e == 't') ? '\t' : e;
            } else if (c == '"') {
                i = j + 1;
                closed = true;
                break;
            } else {
                s += c;
            }
        }
        if (!closed) return false;
        o.push_back(s);
    }
    return false;
}

static bool extract_int(const std::string& line, const std::string& key, int& val) {
    std::string pat = "\"" + key + "\"";
    size_t p = line.find(pat);
    if (p == std::string::npos) return false;
    p = line.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    size_t i = p + 1;
    while (i < line.size() && line[i] == ' ') ++i;
    if (i < line.size() && line[i] == '"') {  // quoted form: "A".."Z" or "0".."3"
        if (i + 1 < line.size() && line[i + 1] >= 'A' && line[i + 1] <= 'Z' && line[i + 2] == '"') {
            val = line[i + 1] - 'A';
            return true;
        }
        size_t q = line.find('"', i + 1);  // digit string, e.g. HF Rowan/hellaswag "3"
        if (q == std::string::npos) return false;
        try {
            size_t len = 0;
            int v = std::stoi(line.substr(i + 1, q - i - 1), &len);
            if (len == 0 || len != q - i - 1) return false;
            val = v;
            return true;
        } catch (...) {
            return false;
        }
    }
    try {
        size_t len = 0;
        int v = std::stoi(line.substr(i), &len);
        if (len == 0) return false;
        val = v;
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<MCTextItem> load_mmlu_jsonl(const std::string& path, size_t* n_skipped) {
    std::vector<MCTextItem> out;
    size_t skipped = 0;
    std::ifstream in(path);
    if (!in) {
        if (n_skipped) *n_skipped = 0;
        return out;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        MCTextItem it;
        int ans = -1;
        if (!extract_string(line, "question", it.context) ||
            !extract_string_array(line, "choices", it.choices) || it.choices.size() < 2 ||
            !extract_int(line, "answer", ans) || ans < 0 || (size_t)ans >= it.choices.size()) {
            ++skipped;
            continue;
        }
        it.answer = ans;
        out.push_back(std::move(it));
    }
    if (n_skipped) *n_skipped = skipped;
    return out;
}

std::vector<MCTextItem> load_hellaswag_jsonl(const std::string& path, size_t* n_skipped) {
    std::vector<MCTextItem> out;
    size_t skipped = 0;
    std::ifstream in(path);
    if (!in) {
        if (n_skipped) *n_skipped = 0;
        return out;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        MCTextItem it;
        int lab = -1;
        bool has_ctx = extract_string(line, "ctx", it.context);
        if (!has_ctx) {
            // Some exports nest context as {"ctx_a": ..., "ctx_b": ...}; join them.
            std::string a, b;
            if (!extract_string(line, "ctx_a", a)) {
                ++skipped;
                continue;
            }
            extract_string(line, "ctx_b", b);
            it.context = b.empty() ? a : (a + " " + b);
        }
        if (!extract_string_array(line, "endings", it.choices) || it.choices.size() < 2 ||
            !extract_int(line, "label", lab) || lab < 0 || (size_t)lab >= it.choices.size()) {
            ++skipped;
            continue;
        }
        it.answer = lab;
        out.push_back(std::move(it));
    }
    if (n_skipped) *n_skipped = skipped;
    return out;
}

EvalResult evaluate(const GPT& model, const std::vector<std::vector<int>>& data) {
    EvalResult res;
    res.mmlu = 0;
    res.hellaswag = 0;
    res.ppl = 0;
    if (data.empty()) {
        return res;
    }
    float total_loss = 0;
    size_t total_tokens = 0;
    size_t correct = 0;
    size_t total_pred = 0;
    for (auto& seq : data) {
        if (seq.empty()) continue;
        // Truncate to block_size if needed
        std::vector<int> batch = seq;
        const auto& cfg = model.config();
        if(batch.size() > cfg.block_size){
            batch.assign(seq.end() - cfg.block_size, seq.end());
        }
        auto logits = model.forward(batch);
        float loss = compute_loss(logits, batch);
        total_loss += loss * batch.size();
        total_tokens += batch.size();
        // Token accuracy for mmlu proxy: predict next token via argmax
        for(size_t i=0;i+1<batch.size() && i+1<logits.shape[0];++i){
            size_t pred = logits.argmax(i);
            int target = batch[i+1];
            // Clamp target to vocab
            int vocab = (int)cfg.vocab_size;
            target = ((target % vocab) + vocab) % vocab;
            if((int)pred == target) correct++;
            total_pred++;
        }
    }
    if (total_tokens > 0) {
        float avg_loss = total_loss / float(total_tokens);
        res.ppl = std::exp(avg_loss);
    }
    if(total_pred>0){
        res.mmlu = float(correct) / float(total_pred);
        // HellaSwag proxy: same as mmlu for now, but could be per-sequence accuracy
        // For true HellaSwag, would compare log-likelihood of choices; we approximate with same accuracy
        res.hellaswag = res.mmlu;
    }
    return res;
}

}  // namespace llm
