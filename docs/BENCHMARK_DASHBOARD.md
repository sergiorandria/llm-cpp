# Benchmark Dashboard


## E48 speculative gate (test_spec_gate, tiny 4-layer target / 2-layer draft, k=4, 12 tokens)
- spec_ms=8.44 greedy_ms=3.28 speedup=0.39 — SLOWER at this scale (draft ~1/2 target, not 1/10).
- The 2x claim in speculative.h holds only when draft << target (10M vs 124M); gate asserts equivalence, timing informational.
Flash block tune: best 128 for head_dim 64, results {32: 11.6796, 64: 11.311, 128: 11.2064, 256: 11.5109}
GEMM tile: M=N=K=64 (anchor {32: 11.6796, 64: 11.311, 128: 11.2064, 256: 11.5109})
