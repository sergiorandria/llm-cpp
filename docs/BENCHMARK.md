# Benchmark

## Matmul 128x128
- Naive: ~45 ms (single-thread)
- Numpy-cpp blocked GEMM + SIMD: ~18 ms (see `bench_matmul`)

Run:
```bash
cmake -B build -DUSE_NUMPY_CPP=ON -DBUILD_TESTS=ON
cmake --build build -j --target bench_matmul && ./build/bench_matmul
```

## Sampling
`top-p` and `top-k` sampling via `src/sampling.cpp` — numpy RNG not needed but `Tensor` ops benefit.

## Training
`Trainer::train` uses `np::linalg::matmul` for every forward; expected 2-3x speedup on 768-dim.
