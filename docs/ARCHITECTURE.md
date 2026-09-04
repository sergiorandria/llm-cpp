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
