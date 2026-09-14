# Task 2 Report: MambaBlock CPU Forward

## Status: DONE

## Files Created/Modified
- `include/llm/mamba.h` — MambaBlock class declaration with 9 parameter tensors
- `src/mamba.cpp` — CPU forward: RMSNorm → in_proj → split gate/SSM → Conv1D+SiLU → x_proj → dt/B/C → selective_scan_sequential → gate → out_proj → residual
- `tests/test_mamba_block.cpp` — unit test (n_embd=32, d_inner=64, d_state=4, dt_rank=4, conv_kernel=4)
- `tests/CMakeLists.txt` — added test_mamba_block target

## Test Results
```
max_diff from residual: 0
parameter tensors: 9
test_mamba_block passed
```

## Notes
- Fixed 3D tensor indexing: `conv1d_weight_(d,0,j)` → `conv1d_weight_.data[d*k+j]` (Tensor only supports 2D `operator()`)
- Fixed 1D tensor indexing: `conv1d_bias_[d]` → `conv1d_bias_.data[d]`
- Backward is a stub (returns zeros) — gradient check deferred to later task

## Addendum: Review Fixes (2026-09-14)

### Issue 1: Test style violation
- Replaced all `assert()` calls in `tests/test_mamba_block.cpp` with `if (fail) { std::cerr << "FAIL: ..."; return 1; }` pattern matching `test_mamba_scan.cpp`
- Added exact parameter count validation: `params.size() != 9` instead of `params.size() > 5`

### Issue 2: Layernorm signature
- Verified `layernorm(const Tensor* gamma, const Tensor* beta, float eps)` in `include/llm/tensor.h`
- Call `x.layernorm(&norm_weight_)` is correct — passes `gamma` only, `beta` and `eps` use defaults

### Issue 3: Unused `conv_state_` member
- Removed `mutable std::vector<float> conv_state_` from `include/llm/mamba.h`
- Removed `conv_state_.assign(...)` from `src/mamba.cpp` constructor

### Issue 4: `assert()` in production code
- Replaced `assert(x.shape[1] == n_embd_)` and `assert(proj.shape[0] == T && ...)` with `if (...) throw std::runtime_error(...)`
- Removed `#include <cassert>` from `src/mamba.cpp`

### Files Modified
- `tests/test_mamba_block.cpp` — test style fixes
- `src/mamba.cpp` — production error handling, removed unused init
- `include/llm/mamba.h` — removed unused member

### Test Results
```
max_diff from residual: 0
parameter tensors: 9
test_mamba_block passed
```
