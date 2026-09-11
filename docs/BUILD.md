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
