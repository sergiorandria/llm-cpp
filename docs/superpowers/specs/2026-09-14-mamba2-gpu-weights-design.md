# Design: MAMBA-2 SSM Architecture + GPU Weight Storage

**Date:** 2026-09-14
**Status:** Approved
**Scope:** Full architecture replacement (Transformer → MAMBA-2) + GPU-resident model weights

---

## 1. Overview

Replace the entire decoder-only Transformer stack (MultiHeadAttention + FeedForward + KVCache) with MAMBA-2 selective state space blocks (Gu & Dao, 2024). Simultaneously, add GPU-resident model weights so all inference matmuls run on-device without PCIe transfers.

**Key properties of MAMBA-2:**
- Linear complexity in sequence length (O(n) vs O(n²) for attention)
- Selective scan: input-dependent gating (B, C, Δ depend on x)
- SSD framework: parallel associative scan (training) or sequential recurrence (inference)
- Recurrent state: O(d_inner × d_state) per layer, constant memory vs sequence length

---

## 2. MAMBA-2 Block Architecture

### 2.1 Block Flow

```
Input x [T, n_embd]
  |
  +--> RMSNorm(x)                              [T, n_embd]
  |
  +--> in_proj: Linear(n_embd, 2*d_inner)      [T, 2*d_inner]
  |     Split into:
  |       x_z   = proj[:, :d_inner]            [T, d_inner]   (gate branch)
  |       x_ssm = proj[:, d_inner:]            [T, d_inner]   (SSM branch)
  |
  +--> x_ssm --> Conv1D(d_inner, kernel=4)     [T, d_inner]
  |              --> SiLU activation
  |
  +--> SSM:
  |     x_proj: Linear(d_inner, dt_rank+2*d_state)  [T, dt_rank+2*d_state]
  |     dt_proj: Linear(dt_rank, d_inner)            [T, d_inner]   (delta)
  |     softplus(dt) -> Δ_t                          (step size)
  |     A_log -> -exp(A_log) -> A                    (state transition, diag)
  |     B = x_proj[:, dt_rank:dt_rank+d_state]       [T, d_state]
  |     C = x_proj[:, dt_rank+d_state:]              [T, d_state]
  |     D = skip connection                           [d_inner]
  |     y = selective_scan(x_ssm, Δ, A, B, C)        [T, d_inner]
  |     y = y + D * x_ssm                             (skip)
  |
  +--> x_z --> SiLU(x_z)                         [T, d_inner]   (gate)
  |
  +--> gate * y --> out_proj: Linear(d_inner, n_embd)  [T, n_embd]
  |
  +--> Residual: out = x + proj_out
  |
Output [T, n_embd]
```

### 2.2 Selective Scan (Core Operation)

The selective scan computes:

```
h_t = A_t * h_{t-1} + B_t * x_t       (state update)
y_t = C_t * h_t + D * x_t             (output)
```

Where A_t, B_t, C_t are input-dependent (selective).

**Training mode (parallel):** Associative scan — O(T log T) via prefix-sum on (A, B) pairs.

**Inference mode (sequential):** Standard recurrence — O(1) per step, O(T) total. State `h` persists across tokens.

### 2.3 Parameters Per Block

| Parameter | Shape | Notes |
|-----------|-------|-------|
| `norm.weight` | `[n_embd]` | RMSNorm gamma |
| `in_proj.weight` | `[2*d_inner, n_embd]` | No bias (MAMBA-2 convention) |
| `conv1d.weight` | `[d_inner, 1, conv_kernel]` | Depth-wise, groups=d_inner |
| `conv1d.bias` | `[d_inner]` | |
| `x_proj.weight` | `[dt_rank + 2*d_state, d_inner]` | Projects to dt, B, C |
| `dt_proj.weight` | `[d_inner, dt_rank]` | Delta projection |
| `A_log` | `[d_inner, d_state]` | Log-space for stability |
| `D` | `[d_inner]` | Skip connection (init=1) |
| `out_proj.weight` | `[n_embd, d_inner]` | No bias |

**Parameter count per block:** ~`2 * n_embd * d_inner + d_inner * (dt_rank + 2*d_state + d_inner) + n_embd * d_inner`
For default `d_inner = 2*n_embd`: approximately `5 * n_embd² + n_embd * d_state`

### 2.4 Default Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `d_inner` | `2 * n_embd` | Inner dimension |
| `d_state` | 16 | SSM state size |
| `dt_rank` | `ceil(n_embd / 16)` | Delta rank (small = cheaper projection) |
| `conv_kernel` | 4 | Local convolution width |
| `dt_min` | 0.001 | Delta clamp minimum |
| `dt_max` | 0.1 | Delta clamp maximum |
| `dt_init_std` | 0.001 | Delta projection init |

---

## 3. GPU Weight Storage

### 3.1 Design

A `GPUWeights` class manages one contiguous GPU allocation per model parameter tensor.

```cpp
class GPUWeights {
public:
    GPUWeights() = default;
    ~GPUWeights();

    // Upload all model parameters to GPU
    void upload(const std::vector<Tensor>& params);

    // Download GPU weights back to CPU tensors
    void download(std::vector<Tensor>& params) const;

    // Device pointer for i-th parameter (for kernel calls)
    float* device_ptr(size_t idx) const;
    const float* device_ptr(size_t idx) const;

    // Number of parameters stored
    size_t count() const;

    // Release all GPU memory
    void clear();

private:
    std::vector<void*> d_ptrs_;
    std::vector<size_t> sizes_;  // element count per tensor
};
```

### 3.2 Integration Points

1. **After model load/init:** `gpu_weights_.upload(model.parameters())`
2. **Forward pass (GPU path):** Kernels read directly from `device_ptr(i)` — no per-call transfer
3. **Backward pass:** Gradients computed on GPU, accumulated in GPU buffers, downloaded for optimizer step
4. **Checkpoint save:** `gpu_weights_.download(...)` before serialization
5. **Quantization:** GPU-side dequantize for inference (Q4/Q8 on-device)

### 3.3 Fallback

When `USE_CUDA` is not defined or no device is present:
- `GPUWeights` is a no-op (empty, all methods return null/0)
- Model runs entirely on CPU as before
- The `use_cuda` config flag controls whether upload is attempted

---

## 4. Recurrent State (Replaces KVCache)

### 4.1 State Structure

```cpp
struct MambaState {
    std::vector<Tensor> h;  // Per-layer hidden state [d_inner, d_state]
    size_t cur_len = 0;
};
```

- `h[layer]` shape: `[d_inner, d_state]` — the recurrent hidden state
- Constant size per layer (unlike KV-cache which grows with sequence length)
- Total state: `n_layers * d_inner * d_state * sizeof(float)` bytes

### 4.2 Update

```cpp
void MambaState::update(size_t layer, const Tensor& x, /* ssm params */) {
    // Sequential scan: h = A * h + B * x, y = C * h
    // x is [1, d_inner] (single token)
    // h[layer] is [d_inner, d_state]
}
```

### 4.3 GPU State

The recurrent state can also be kept on GPU for true GPU inference:
```cpp
std::vector<void*> d_h_;  // device pointers for h per layer
```

---

## 5. Config Changes

### 5.1 New Fields in `Config`

```cpp
// MAMBA-2 architecture (when use_mamba=true, replaces Transformer)
bool use_mamba = false;        // true = MAMBA-2 blocks
size_t d_inner = 0;            // 0 = auto: 2 * n_embd
size_t d_state = 16;           // SSM state dimension
size_t dt_rank = 0;            // 0 = auto: ceil(n_embd / 16)
size_t conv_kernel = 4;        // Conv1D kernel width
bool use_gpu_weights = false;  // true = upload weights to GPU
```

### 5.2 Auto-Configuration

When `use_mamba = true`:
- `d_inner = 2 * n_embd` if not set
- `dt_rank = ceil(n_embd / 16)` if not set
- `pos_encoding` is ignored (MAMBA doesn't use positional encodings — the state captures position implicitly)
- `n_heads` is ignored (no attention)
- `weight_tying` still applies

---

## 6. Serialization

### 6.1 Binary Format (v4)

The checkpoint format changes to accommodate MAMBA block parameters:

```
[4 bytes] magic = 0x4C4C4D00
[4 bytes] version = 4
[8 bytes] vocab_size, n_layers, n_heads, n_embd, block_size
[8 bytes] d_inner, d_state, dt_rank, conv_kernel, use_mamba

-- Global tensors (same as v3):
  wte_, wpe_, ln_f_gamma_, ln_f_beta_, lm_head_

-- Per-block tensors (order changes for MAMBA):
  if use_mamba:
    norm.weight
    in_proj.weight
    conv1d.weight, conv1d.bias
    x_proj.weight
    dt_proj.weight
    A_log
    D
    out_proj.weight
  else:
    (Transformer tensors as before — backward compatible v3 path)
```

### 6.2 GGUF Format

Same GGUF v3 container, different tensor names:
- `block_i.norm.weight`, `block_i.in_proj.weight`, etc.
- Auto-detect MAMBA vs Transformer from tensor names on load

---

## 7. Backward Pass

MAMBA-2 backward requires the selective scan adjoint:

```
dh_t = C_t^T * dy_t + A_{t+1} * dh_{t+1}    (state gradient)
dx_t = B_t^T * dh_t + D * dy_t                (input gradient)
dA_t = diag(h_{t-1} * dh_t)                   (A gradient)
dB_t = x_t * dh_t^T                            (B gradient)
dC_t = h_t * dy_t^T                            (C gradient)
ddt_t = ... (via dt_proj)
```

The parallel scan for backward uses the same associative scan technique, applied in reverse.

---

## 8. Testing Strategy

| Test | Purpose |
|------|---------|
| `test_mamba_scan.cpp` | Selective scan: parallel vs sequential parity |
| `test_mamba_block.cpp` | MambaBlock forward correctness + gradient check |
| `test_mamba_model.cpp` | Full model: forward, generate, incremental parity |
| `test_gpu_weights.cpp` | Upload/download parity, fallback behavior |
| `test_mamba_overfit.cpp` | Small model overfits tiny dataset |
| `test_mamba_checkpoint.cpp` | Save/load roundtrip bit-exact |

---

## 9. Files to Create/Modify

### New Files
- `include/llm/mamba.h` — MambaBlock class declaration
- `src/mamba.cpp` — MambaBlock forward/backward (CPU)
- `include/llm/mamba_scan.h` — Selective scan interface
- `src/mamba_scan.cpp` — Parallel + sequential scan (CPU)
- `src/mamba_scan.cu` — CUDA selective scan kernels
- `include/llm/gpu_weights.h` — GPU weight manager
- `src/gpu_weights.cu` — CUDA memory management
- `include/llm/mamba_state.h` — Recurrent state class
- `src/mamba_state.cpp` — State management
- `tests/test_mamba_scan.cpp`
- `tests/test_mamba_block.cpp`
- `tests/test_mamba_model.cpp`
- `tests/test_gpu_weights.cpp`

### Modified Files
- `include/llm/model.h` — Config Mamba fields, GPT uses MambaBlock
- `src/model.cpp` — Forward/backward/selective_scan dispatch
- `CMakeLists.txt` — New source files, CUDA kernel compilation
- `tests/CMakeLists.txt` — New test targets

### Removed Files (or no longer compiled)
- `include/llm/attention.h` / `src/attention.cpp` — Replaced by MAMBA
- `include/llm/feed_forward.h` / `src/feed_forward.cpp` — Replaced by MAMBA
- `include/llm/transformer.h` / `src/transformer.cpp` — Replaced by MAMBA
- `include/llm/flash_attention.h` / `src/flash_attention.cpp` — No longer needed
- `include/llm/kv_cache.h` / `src/kv_cache.cpp` — Replaced by MambaState
- `include/llm/alibi.h` / `src/alibi.cpp` — No positional encoding needed
- `include/llm/sliding.h` / `src/sliding.cpp` — No sliding window needed
- `include/llm/mqa.h` / `src/mqa.cpp` — No MQA (no attention)
- Related tests for removed modules

---

## 10. Migration Notes

- Old Transformer checkpoints (v3) will NOT load into MAMBA models
- The `use_mamba` flag in config.json distinguishes architectures
- GGUF files can store either Transformer or MAMBA tensors
- The C API (`llm-c-api`) gains new functions for MAMBA inference
- Python bindings continue to work (model interface unchanged)
