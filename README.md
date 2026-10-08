# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230103252  
**Allocated GPU Node:** Tesla T4 (Google Colab, 15360 MiB, 40 SMs)  
**CUDA Compute Capability:** 7.5  
**Official Verification Token:** 358B68BDE78A5148BE9B

## How to run (Google Colab, T4 GPU)

```
!git clone https://github.com/ItachiTore/cuda-lab-02-230103252.git
%cd cuda-lab-02-230103252
!python task1_divergence.py
!python task2_stencil_1d.py
!python task3_grid_stride.py
!python task4_sobel_2d.py
!echo 230103252 | python verify_submission.py
```

## Task 1 — Warp Divergence Microbenchmark

N = 2^20 = 1,048,576 float32, 1,000 iterations per element, grid 4096 × 256.
Kernel-only time (data stays in VRAM), 1 warm-up launch + mean of 10 trials (`cuda.synchronize()` + `time.perf_counter()`).

- Path 1 (multiply-accumulate): `y = y * 1.0001 + 0.0001`
- Path 2 (subtract-divide): `y = (y - 0.0001) / 1.0001`

| Kernel | Branch condition | Mean time (ms) | Std (ms) | vs A |
|---|---|---:|---:|---:|
| A: Uniform Path | none (all Path 1) | 2.944 | 0.024 | 1.00× |
| B: Full Divergence (interleaved) | `idx % 2 == 0` | 9.464 | 0.022 | 3.21× |
| C: Warp-Aligned Branching | `(idx // 32) % 2 == 0` | 3.691 | 0.018 | 1.25× |

**Divergence penalty B vs C: 2.56×.** Both kernels do exactly the same amount of work (half the elements take
Path 1, half take Path 2); only the placement of the branch differs.

**Analysis**
- In **B** every warp contains both even and odd threads, so the SM executes Path 1 with odd lanes masked off and
  then Path 2 with even lanes masked off. Each warp pays for *both* loops: ≈ T(Path 1) + T(Path 2).
- In **C** the condition is constant across each warp of 32 threads, so every warp runs only one path at full
  width. There is no serialization; the time is just the average of the two paths.
- C is 1.25× slower than A not because of divergence but because Path 2 is more expensive: an IEEE float division is
  a multi-instruction sequence, unlike a single FMA. From C ≈ (A + P2) / 2 we get P2 ≈ 4.4 ms. So for B the ideal
  serialized estimate is A + P2 ≈ 7.4 ms. The measured 9.46 ms is even higher because each half-masked warp also
  loses the latency hiding that full warps provide.
- Takeaway: branch on data that is uniform per warp (e.g. `warp_id`, `blockIdx`), not on `threadIdx % k`.

## Task 2 — 1D Stencil with Boundary Clamping

`y[i] = 0.25·x[i-1] + 0.5·x[i] + 0.25·x[i+1]`, with halo replication: `x[-1] := x[0]`, `x[N] := x[N-1]`.

| N | Launch | Idle guarded threads | Max \|GPU − CPU\| | Result |
|---|---|---:|---:|---|
| 100,007 (odd, non-power-of-two) | 391 × 256 | 89 | 2.384e-07 | `TASK 2 PASSED` |

The `if idx < N` guard stops the 89 surplus threads of the last block. Clamping the indices at `idx == 0` and
`idx == N-1` replaces the out-of-bounds reads with the edge value, which is exactly `np.pad(..., mode='edge')`.
Edge check: `out[0] = -0.685391` and `out[N-1] = 0.591748`, identical on the CPU.
The delta (~2.4e-7) is float32 rounding: the CPU reference is computed in float64.

## Task 3 — Grid-Stride Loop

| N | Launch | Threads | Elements / thread | Roundtrip (H2D + kernel + D2H) | Result |
|---|---|---:|---:|---:|---|
| 16,777,216 (2^24) | 64 × 256 | 16,384 | 1,024 | 39.655 ms | all elements == 4.25, 0 mismatches |

Each thread starts at `cuda.grid(1)` and jumps by `cuda.gridsize(1)` = 16,384, so a fixed, small grid covers
any N with no extra boundary logic: the `range(start, N, stride)` loop is its own bounds check. Consecutive
threads still touch consecutive addresses in each iteration, so global-memory accesses stay coalesced.
Numba warns that 64 blocks under-fill the 40 SMs of the T4. That limit is the constraint this task imposes, and
the roundtrip is dominated by the 2 × 64 MB PCIe transfers anyway.

## Task 4 — 2D Sobel-X

```
K_x = [[-1, 0, +1],
       [-2, 0, +2],
       [-1, 0, +1]]
```

| Input | Block | Grid | Total threads | Max \|GPU − CPU\| | Borders | Result |
|---|---|---|---:|---:|---|---|
| 2048 × 2048 float32 | 16 × 16 | 128 × 128 = 16,384 blocks | 4,194,304 | 4.768e-07 | all 0.0 | `TASK 4 PASSED` |

`cuda.grid(2)` returns `(x, y)`, so x maps to **columns** and y to **rows**. The grid is
`(ceil(cols/16), ceil(rows/16))`. Interior pixels apply the 3×3 kernel. Pixels on the one-pixel border are
written as 0.0 explicitly, because `device_array` memory is not zero-initialised.
Extra sanity check: on a horizontal ramp (x-gradient = 1) every interior value is 8.0. That is the expected value,
since the weights on the +1 side sum to 1 + 2 + 1 = 4 and the difference across the kernel spans 2 pixels.

## Verification

```
[PASS] Task 2 (1D Stencil & Clamping)
[PASS] Task 3 (Grid-Stride Scaling)
[PASS] Task 4 (2D Sobel Horizontal)
VERIFICATION SUCCESSFUL
OFFICIAL SUBMISSION TOKEN: 358B68BDE78A5148BE9B
```
