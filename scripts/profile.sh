#!/bin/bash
set -e
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_NUMPY_CPP=ON
cmake --build build -j --target bench_matmul
perf record -g ./build/bench_matmul 2>/dev/null || ./build/bench_matmul
echo "profile done"
