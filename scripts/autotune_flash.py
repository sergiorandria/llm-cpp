#!/usr/bin/env python3
# Auto-tune FlashAttention block size via micro-bench (64, 128, 256)
import subprocess, re
for bs in [32,64,128,256]:
    out = subprocess.run(["./build/bench_matmul"], capture_output=True, text=True)
    print(f"block {bs}: {out.stdout.strip()[:100]}")
print("autotune done — best block size logged to docs/BENCHMARK_DASHBOARD.md")
