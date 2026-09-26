# Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DENABLE_SANITIZERS=OFF
cmake --build build -j
ctest --test-dir build
./build/llm-cpp --help
./build/llm-cpp train --config config/train_tinystories.json
./build/llm-cpp generate --prompt "Hello" --temperature 0.8
```

## Threads (I81)

GEMM/softmax/norm loops use OpenMP when available. Control at runtime:

```bash
LLM_THREADS=4 ./build/llm-cpp generate --prompt "Hi"   # preferred
OMP_NUM_THREADS=4 ./build/llm-cpp serve --port 8080    # fallback
```

`init_threading()` prefers `LLM_THREADS`, else `OMP_NUM_THREADS`, else build
default. Expect ≥1.5× 1→4 threads on 512² `bench_matmul`; pin with
`OMP_PROC_BIND=spread OMP_PLACES=cores` on NUMA boxes. Deterministic mode
(`TrainConfig::deterministic`) forces 1 thread.

> I89 finding: on SMALL models (≤64 embd) nested OpenMP (our loops × numpy-cpp
> micro-GEMMs) oversubscribes — 12 threads measured **~160× slower** (0.37 vs
> 59 tok/s), reproduced on bench box at 87× (12 vs 1078 tok/s).
> `init_threading()` now caps nesting at 1 by default (inner kernels run
> serially, outer parallelism kept) — unpinned bench went 12 → 452 tok/s.
> Override with `OMP_MAX_ACTIVE_LEVELS`/`OMP_NESTED` if you know better.
> For max small-model throughput still pin `OMP_NUM_THREADS=1` (1078 tok/s);
> the bench gate (`.github/workflows/bench.yml`) pins threads=1 for this reason.
