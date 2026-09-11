// G63: stable C ABI for Python (ctypes) and other FFIs. No pybind11 needed.
#include "llm/config.h"
#include "llm/model.h"
#include "llm/tokenizer.h"
#include <cstring>
#include <string>
#include <vector>

struct llm_model {
    llm::GPT* gpt;
    llm::Tokenizer* tok;
};

extern "C" {

llm_model* llm_create(size_t vocab, size_t layers, size_t heads, size_t embd, size_t block) {
    llm::Config c;
    c.vocab_size = vocab; c.n_layers = layers; c.n_heads = heads;
    c.n_embd = embd; c.block_size = block;
    if (!llm::validate_config(c)) return nullptr;
    auto* m = new llm_model{new llm::GPT(c), new llm::Tokenizer(256)};
    return m;
}

void llm_free(llm_model* m) {
    if (!m) return;
    delete m->gpt;
    delete m->tok;
    delete m;
}

// Encodes UTF-8 text -> ids. Returns count (or needed count if out==null/cap==0).
int llm_encode(llm_model* m, const char* text, int* out, int cap) {
    if (!m || !text) return -1;
    auto ids = m->tok->encode(text);
    if (!out || cap <= 0) return (int)ids.size();
    int n = (int)std::min<size_t>(ids.size(), (size_t)cap);
    for (int i = 0; i < n; ++i) out[i] = ids[i];
    return n;
}

// Decodes ids -> UTF-8 bytes. Returns bytes written (or needed if out==null/cap==0).
int llm_decode(llm_model* m, const int* ids, int n, char* out, int cap) {
    if (!m || (!ids && n > 0)) return -1;
    std::vector<int> v(ids, ids + n);
    std::string s = m->tok->decode(v);
    if (!out || cap <= 0) return (int)s.size();
    int k = (int)std::min<size_t>(s.size(), (size_t)cap);
    memcpy(out, s.data(), k);
    return k;
}

// Greedy generate: returns total ids written (prompt + new) or needed count.
int llm_generate(llm_model* m, const char* prompt, int max_tokens, int* out, int cap) {
    if (!m || !prompt || max_tokens < 0) return -1;
    auto ids = m->tok->encode(prompt);
    auto gen = m->gpt->generate(ids, (size_t)max_tokens, 0.0f, 0);
    if (!out || cap <= 0) return (int)gen.size();
    int n = (int)std::min<size_t>(gen.size(), (size_t)cap);
    for (int i = 0; i < n; ++i) out[i] = gen[i];
    return n;
}

size_t llm_num_parameters(const llm_model* m) {
    if (!m) return 0;
    return m->gpt->num_parameters();
}

} // extern "C"
