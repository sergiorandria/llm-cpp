# Architecture

Decoder-only Transformer as in GPT-2.

```
Tokens -> WTE + WPE -> N x TransformerBlock -> LN -> LM Head
```

Each block: Pre-LN -> Attention (causal) -> Residual -> Pre-LN -> FFN -> Residual

- Tensor: `Tensor::matmul` dispatches to `np::linalg::matmul` when `USE_NUMPY_CPP`, `randn` via `np::random::Generator` PCG64
- Attention: true per-head split `src/attention.cpp:14` (`slice_head` loop, `Qh·Kh^T/sqrt(head_dim)`, causal, dropout), `n_heads` now affects output (verified `test_attention_heads`)
- FFN: GELU + SwiGLU gated `src/transformer.cpp:17`, layernorm with `gamma/beta` `src/transformer.cpp:46`
- PosEncoding: Learned, Sinusoidal `src/model.cpp:22`, RoPE `src/model.cpp:36` applied in `forward_with_hidden`
- KV-cache: unified `KVCache` `src/model.cpp:72` (instantiated in `generate`, `update` last hidden), still `O(n²)` recompute until per-layer cache wired — documented as `[~]` in README

## Backend
`llm::Tensor` is the acceleration point: `to_ndarray()`/`from_ndarray()` bridge to `np::ndarray<float>`. All GEMMs go through `np::linalg::matmul` (blocked, SIMD, threading). See `docs/NUMPY_BACKEND.md`.
