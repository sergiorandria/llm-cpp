# Training

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/llm-cpp train --config config/train_tinystories.json --data data/input.txt
```

Trainer uses AdamW + cosine scheduler + grad clipping.
Loss: cross-entropy.
Checkpoint: binary v1 format (magic 0x4C4C4D00)

## Optimizers
| Optim | LR | Weight Decay |
|---|---|---|
| SGD | 0.01 | 0 |
| Adam | 1e-3 | 0 |
| AdamW | 6e-4 | 0.1 |
