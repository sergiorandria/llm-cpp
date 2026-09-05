# Continuous Optimization Cycle

This project never stops — every finished task immediately spawns the next.

**Current cycle (2026-09-05 08:21):**
- Cycle 14: Vocabulary pruning + dynamic block size (context extension to 2048 via RoPE scaling)
- Cycle 15: 4-bit GPTQ quantization (4× memory, 0.3 PPL loss)
- Cycle 16: Pipeline parallel micro-batch (n_layers=12 → 3 stages)

**Previous cycles (completed):**
- 2026-09-04: 137 commits — 100 tasks + numpy backend + BPE bug + MHA split + trainer + checkpoint v3
- 2026-09-05 07:xx: 14 tests, KVCache per-layer fix, scheduler LR, fused kernels, FlashAttention, LoRA, benchmark dashboard
- 2026-09-05 08:xx: activation checkpointing, prefetch, speculative decoding, fuzz, SoA, OpenBLAS, per-layer dropout, autotune

**Rule:** When `TodoWrite` shows all `completed`, immediately create next `TodoWrite` with new cycles. No idle.

**Next auto-cycle will be:** Cycle 17-20 (MoE, RLHF, distill, eval harness) — triggered when Cycle 14-16 done.
