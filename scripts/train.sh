#!/bin/bash
set -e
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/llm-cpp train --config config/train_tinystories.json --data data/input.txt
