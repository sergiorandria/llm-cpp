# Architecture

Decoder-only Transformer as in GPT-2.

```
Tokens -> WTE + WPE -> N x TransformerBlock -> LN -> LM Head
```

Each block: Pre-LN -> Attention (causal) -> Residual -> Pre-LN -> FFN -> Residual

- Tensor: naive CPU matmul, softmax, layernorm
- Attention: single-head skeleton, multi-head split TODO
- FFN: GELU, SwiGLU optional
- PosEncoding: Learned, Sinusoidal, RoPE

## Backend
`llm::Tensor` is the acceleration point: `to_ndarray()`/`from_ndarray()` bridge to `np::ndarray<float>`. All GEMMs go through `np::linalg::matmul` (blocked, SIMD, threading). See `docs/NUMPY_BACKEND.md`.
