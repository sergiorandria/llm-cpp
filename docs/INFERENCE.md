# Inference

```bash
./build/llm-cpp generate --checkpoint checkpoints/model.bin --prompt "Hello" --temperature 0.8 --top_k 40 --top_p 0.9
```

Features: KV-cache, temperature, top-k, top-p, repetition penalty, streaming.

KV-cache reduces O(T^2) to O(T) per token.
