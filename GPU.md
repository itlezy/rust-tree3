# GPU Execution Feasibility Notes

## Question

Could the tree sequence search run entirely on the GPU and its VRAM, like a video game?

---

## What Maps Well to GPU

- **Fingerprint compatibility check** (`TreeFingerprint::compatible`) — 17-byte struct comparison across all N candidates simultaneously. Pure data-parallel, no branching. Ideal GPU workload.
- **Rejection bitset scan** (`find_first_live`) — reading N `AtomicBool`s and finding the first unset one is a parallel reduction. GPUs do this fast.
- **Post-acceptance sweep** (`CandidatePool::sweep`) — currently uses rayon (CPU threads); the same pattern maps naturally to a GPU compute dispatch. Each candidate is independent.

The two hot flat arrays (`fingerprints`, `rejected`) are already contiguous, fixed-size, and layout-friendly for GPU memory.

---

## What Maps Badly to GPU

- **Recursive backtracking** (`embeds` / `match_children`) — irregular, variable-depth recursion with dynamic branching per tree pair. GPUs execute in lockstep warps; divergent control flow causes most threads to stall. This is the worst possible workload for a GPU.
- **Dynamic heap allocation** — `Tree` clones use `Vec<Node>` (heap-allocated, variable size). GPU memory has no malloc equivalent per thread.
- **Tree generation phase** (`gen_combos_cached`) — deeply recursive partition enumeration with a shared mutable memoization cache. Fundamentally sequential; no GPU mapping.

---

## Realistic Hybrid Architecture

Keep CPU responsible for:
- Tree generation and canonicalization
- The recursive `embeds` check (backtracking)
- The memoization cache

Offload to GPU:
- Fingerprint pre-filter sweep over all N candidates (GPU compute shader)
- Rejection bitset compaction / scan

Libraries that could enable this in Rust:
- [`wgpu`](https://docs.rs/wgpu/latest/wgpu/) — cross-platform GPU compute (WebGPU API); no CUDA required
- [`cudarc`](https://docs.rs/cudarc/latest/cudarc/) — CUDA kernels from Rust; NVIDIA only
- [`opencl3`](https://docs.rs/opencl3/latest/opencl3/) — OpenCL; cross-vendor

---

## Full GPU Path (Research-Level)

Running the embedding check on GPU would require:

1. **Iterative stack-based embedding** — rewrite `match_children` as an explicit stack (no recursion; GPU shaders have no call stack).
2. **Flat tree encoding** — trees are already `Vec<Node>` with index-based children; this is the right format. Would need to pack into a GPU buffer as a struct-of-arrays.
3. **GPU-side work queue** — a persistent kernel with a device-side queue dispatching embedding checks as candidates are found live.
4. **Warp divergence mitigation** — group candidates by tree size/shape before dispatching to reduce branch divergence within a warp.

This is feasible as a research project but is a substantial rewrite. The embedding checker would need to be redesigned from scratch for SIMT execution.

---

## Summary Table

| Component | GPU feasible? | Notes |
|---|---|---|
| Fingerprint sweep | Yes, straightforward | Flat array, 17-byte compare, no branching |
| Rejection bitset scan | Yes | Parallel reduction |
| `embeds` recursive check | No without major rewrite | Divergent recursion, dynamic allocation |
| Tree generation | No | Sequential recursive enumeration |
| Full GPU execution | Possible (research) | Requires iterative embedding + flat encoding |
