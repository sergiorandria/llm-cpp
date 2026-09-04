# Contributing

Welcome! This project implements an LLM from scratch in C++.

## Workflow
- Fork, create feature branch, PR to `main`
- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`
- Ensure `cmake -B build && cmake --build build -j && ./build/llm-cpp` passes
- Add tests for new ops

## Code Style
- C++17, clang-format
- Headers in `include/llm/`, impl in `src/`
- No external deps except STL + OpenMP

## Roadmap
See README.md Roadmap. Pick an unchecked item.
