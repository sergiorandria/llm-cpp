#include "llm/tokenizer.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <algorithm>

namespace llm {

Tokenizer::Tokenizer(size_t vocab_size) : vocab_size_(vocab_size) {
    // init byte-level vocab 0-255 mapped (GPT-2 style)
    for(int i=0;i<256 && i<(int)vocab_size_; ++i){
        std::string s(1, char(i));
        vocab_[s]=i;
        inv_vocab_[i]=s;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    // BPE encode: start from char ids, then apply merges greedily
    std::vector<std::string> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text) tokens.emplace_back(1, char(c));
    // apply merges in order learned
    for (auto &mer: merges_) {
        std::string merged = mer.first + mer.second;
        for (size_t i = 0; i + 1 < tokens.size(); ) {
            if (tokens[i] == mer.first && tokens[i+1] == mer.second) {
                tokens[i] = merged;
                tokens.erase(tokens.begin() + i + 1);
            } else ++i;
        }
    }
    std::vector<int> ids;
    ids.reserve(tokens.size());
    for (auto &tok : tokens) {
        auto it = vocab_.find(tok);
        if (it != vocab_.end()) ids.push_back(it->second);
        else ids.push_back(UNK % (int)vocab_size_);
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string s;
    for (int id : ids) {
        auto it = inv_vocab_.find(id);
        if (it != inv_vocab_.end()) s += it->second;
        else if (id >=0 && id < 256) s.push_back(char(id));
        else s.push_back('?');
    }
    return s;
}

void Tokenizer::load(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    while(std::getline(in, line)){
        if(line.empty() || line[0]=='#') continue;
        size_t sp = line.find(' ');
        if(sp==std::string::npos) continue;
        std::string a=line.substr(0,sp), b=line.substr(sp+1);
        merges_.emplace_back(a,b);
        std::string merged=a+b;
        if(vocab_.find(merged)==vocab_.end() && vocab_.size()<vocab_size_){
            int id=(int)vocab_.size();
            vocab_[merged]=id;
            inv_vocab_[id]=merged;
        }
    }
}

void Tokenizer::train(const std::string& text, size_t num_merges){
    // Real BPE: iteratively merge most frequent adjacent pair (byte-level, UTF-8 safe)
    std::vector<std::string> words;
    for (unsigned char c: text) words.emplace_back(1, char(c));
    // naive but correct for small text
    for(size_t iter=0; iter<num_merges && vocab_.size()<vocab_size_; ++iter){
        std::unordered_map<std::string,int> freq;
        for(size_t i=0;i+1<words.size();++i){
            // Use Unit Separator \x1F (not NUL) — NUL would truncate std::string from C-string literal
            const std::string SEP = std::string(1, '\x1F');
            std::string pair = words[i] + SEP + words[i+1];
            freq[pair]++;
        }
        if(freq.empty()) break;
        auto best = std::max_element(freq.begin(), freq.end(),
            [](auto &a, auto &b){return a.second < b.second;});
        const std::string SEP = std::string(1, '\x1F');
        size_t sep = best->first.find(SEP);
        std::string a = best->first.substr(0, sep);
        std::string b = best->first.substr(sep + SEP.size());
        std::string merged = a + b;
        merges_.emplace_back(a,b);
        if(vocab_.find(merged)==vocab_.end()){
            int id=(int)vocab_.size();
            vocab_[merged]=id; inv_vocab_[id]=merged;
        }
        // apply merge to words
        for(size_t i=0;i+1<words.size();){
            if(words[i]==a && words[i+1]==b){ words[i]=merged; words.erase(words.begin()+i+1); }
            else ++i;
        }
    }
}
void Tokenizer::save(const std::string& path) const {
    std::ofstream out(path);
    for(auto &m: merges_) out<<m.first<<" "<<m.second<<"\n";
}

// ── B12: HF compat ──────────────────────────────────────────────
// vocab.json minimal parser: {"tok": id, ...} with \" and \\ escapes (tokens are
// byte strings, no \u needed for our byte-level vocab; \uXXXX handled as raw).
static std::string json_unescape(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char e = s[++i];
            if (e == 'n') o += '\n'; else if (e == 't') o += '\t';
            else if (e == '"') o += '"'; else if (e == '\\') o += '\\';
            else { o += '\\'; o += e; }
        } else o += s[i];
    }
    return o;
}
static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n"; else if (c == '\t') o += "\\t";
        else o += c;
    }
    return o;
}

void Tokenizer::load_hf(const std::string& vocab_path, const std::string& merges_path) {
    // vocab.json
    {
        std::ifstream in(vocab_path);
        std::string j((std::istreambuf_iterator<char>(in)), {});
        vocab_.clear(); inv_vocab_.clear();
        size_t i = 0;
        while (i < j.size()) {
            size_t q1 = j.find('"', i);
            if (q1 == std::string::npos) break;
            size_t q2 = q1 + 1;
            std::string key;
            while (q2 < j.size()) {
                if (j[q2] == '\\' && q2 + 1 < j.size()) { key += j[q2]; key += j[q2+1]; q2 += 2; }
                else if (j[q2] == '"') break;
                else key += j[q2++];
            }
            size_t colon = j.find(':', q2);
            if (colon == std::string::npos) break;
            size_t nstart = j.find_first_of("-0123456789", colon);
            size_t nend = j.find_first_not_of("0123456789", nstart);
            int id = std::stoi(j.substr(nstart, nend - nstart));
            std::string tok = json_unescape(key);
            vocab_[tok] = id; inv_vocab_[id] = tok;
            if (id >= (int)vocab_size_) vocab_size_ = id + 1;
            i = nend;
        }
    }
    // merges.txt (skip #version line and empty lines)
    {
        std::ifstream in(merges_path);
        std::string line;
        merges_.clear();
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            // split on last space? tokens contain no spaces in our byte-level merges
            // except the "Ġ"-style leading space encoded as "Ġ"? keep simple: first space
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            merges_.emplace_back(line.substr(0, sp), line.substr(sp + 1));
        }
    }
}

void Tokenizer::save_hf(const std::string& dir) const {
    // dir must exist; write vocab.json + merges.txt
    {
        std::ofstream out(dir + "/vocab.json");
        out << "{";
        bool first = true;
        // invert sorted by id for determinism
        std::vector<std::pair<int,std::string>> by_id;
        for (auto& kv : inv_vocab_) by_id.emplace_back(kv.first, kv.second);
        std::sort(by_id.begin(), by_id.end());
        for (auto& p : by_id) {
            if (!first) out << ",";
            out << "\n  \"" << json_escape(p.second) << "\": " << p.first;
            first = false;
        }
        out << "\n}\n";
    }
    {
        std::ofstream out(dir + "/merges.txt");
        out << "#version: 0.2\n";
        for (auto& m : merges_) out << m.first << " " << m.second << "\n";
    }
}

// ── B13: streaming decode ───────────────────────────────────────
static size_t utf8_need(const unsigned char* p, size_t avail) {
    // returns total bytes needed for sequence starting at p, 0 if invalid lead
    unsigned char c = p[0];
    size_t need = 0;
    if ((c & 0x80) == 0) return 1;
    else if ((c & 0xE0) == 0xC0) need = 2;
    else if ((c & 0xF0) == 0xE0) need = 3;
    else if ((c & 0xF8) == 0xF0) need = 4;
    else return 1; // invalid lead: emit as-is (byte-level compat)
    if (avail < need) return need; // incomplete
    for (size_t k = 1; k < need; ++k)
        if ((p[k] & 0xC0) != 0x80) return 1; // broken sequence: emit lead byte
    return need;
}

std::string Tokenizer::decode_incremental(const std::vector<int>& ids, std::string& carry) const {
    std::string raw = carry + decode(ids);
    carry.clear();
    // hold back trailing incomplete sequence (max 3 bytes)
    size_t n = raw.size();
    size_t start = n > 4 ? n - 4 : 0;
    for (size_t i = start; i < n; ++i) {
        size_t need = utf8_need((const unsigned char*)raw.data() + i, n - i);
        if (need > n - i) { // incomplete tail at i — is it a real lead?
            unsigned char c = raw[i];
            bool is_lead = (c & 0x80) == 0 || (c & 0xE0) == 0xC0 ||
                           (c & 0xF0) == 0xE0 || (c & 0xF8) == 0xF0;
            // only hold back if everything from i looks like a prefix (continuations valid so far)
            bool prefix_ok = true;
            for (size_t k = i + 1; k < n; ++k)
                if ((((unsigned char)raw[k]) & 0xC0) != 0x80) { prefix_ok = false; break; }
            if (is_lead && prefix_ok) {
                carry = raw.substr(i);
                return raw.substr(0, i);
            }
        }
    }
    return raw;
}

// ── B14: GPT-2 pre-tokenizer (ASCII approx) ─────────────────────
std::vector<std::string> Tokenizer::split_pretokenize(const std::string& text) {
    std::vector<std::string> out;
    size_t i = 0, n = text.size();
    auto is_letter = [](unsigned char c){ return (c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>=0x80); };
    auto is_digit = [](unsigned char c){ return c>='0'&&c<='9'; };
    while (i < n) {
        // contractions: 's 't 're 've 'm 'll 'd
        if (text[i] == '\'' && i + 1 < n) {
            static const char* conts[] = {"s","t","re","ve","m","ll","d"};
            bool hit = false;
            for (auto c : conts) {
                size_t L = strlen(c);
                if (text.compare(i + 1, L, c) == 0) {
                    out.push_back(text.substr(i, 1 + L));
                    i += 1 + L; hit = true; break;
                }
            }
            if (hit) continue;
        }
        if (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r') {
            size_t j = i;
            while (j < n && (text[j]==' '||text[j]=='\t'||text[j]=='\n'||text[j]=='\r')) ++j;
            out.push_back(text.substr(i, j - i));
            i = j; continue;
        }
        // optional single leading space then letters/digits/other
        size_t j = i;
        if (text[j] == ' ') ++j; // at most one leading space attaches
        if (j < n && is_letter((unsigned char)text[j])) {
            ++j; while (j < n && is_letter((unsigned char)text[j])) ++j;
            out.push_back(text.substr(i, j - i)); i = j; continue;
        }
        if (j < n && is_digit((unsigned char)text[j])) {
            ++j; while (j < n && is_digit((unsigned char)text[j])) ++j;
            out.push_back(text.substr(i, j - i)); i = j; continue;
        }
        // other: run of non-space non-alnum (keeps punctuation together, splits high bytes singly? no—group)
        ++j;
        while (j < n && text[j]!=' ' && text[j]!='\t' && text[j]!='\n' && text[j]!='\r'
               && !is_letter((unsigned char)text[j]) && !is_digit((unsigned char)text[j])
               && text[j]!='\'') ++j;
        out.push_back(text.substr(i, j - i)); i = j;
    }
    return out;
}

} // namespace llm
