# Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DENABLE_SANITIZERS=OFF
cmake --build build -j
ctest --test-dir build
./build/llm-cpp --help
./build/llm-cpp train --config config/train_tinystories.json
./build/llm-cpp generate --prompt "Hello" --temperature 0.8
```
