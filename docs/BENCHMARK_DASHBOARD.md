# Benchmark Dashboard


## E48 speculative gate (test_spec_gate, tiny 4-layer target / 2-layer draft, k=4, 12 tokens)
- spec_ms=8.44 greedy_ms=3.28 speedup=0.39 — SLOWER at this scale (draft ~1/2 target, not 1/10).
- The 2x claim in speculative.h holds only when draft << target (10M vs 124M); gate asserts equivalence, timing informational.
Flash block tune: best 128 for head_dim 64, results {32: 11.6796, 64: 11.311, 128: 11.2064, 256: 11.5109}
GEMM tile: M=N=K=64 (anchor {32: 11.6796, 64: 11.311, 128: 11.2064, 256: 11.5109})

## I89 infer baseline (bench_infer --tokens 20 --trials 5, OMP_NUM_THREADS=1, tiny 2L/64e)
- tokens/sec=352.6 ms/token=2.84 total_p50_ms=56.7 total_p95_ms=62.1 caps=scalar+OpenMP+numpy-cpp
- Gate: bench.yml fails on >10% regression vs 352.6. NOTE: default threads give 0.37 tok/s (oversubscription); always pin threads=1 for small models.
