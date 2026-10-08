"""Task 1: Warp divergence microbenchmark.

Three kernels do 1,000 dependent iterations per element on N = 2^20 float32 values:
  A - uniform path (every thread runs Path 1)
  B - interleaved divergence (idx % 2 picks the path -> both paths inside every warp)
  C - warp-aligned branching (warp_id % 2 picks the path -> no intra-warp divergence)
Only kernel execution is timed (data stays in VRAM): 1 warm-up + mean of 10 trials.
"""
import time

import numpy as np
from numba import cuda

N = 1 << 20          # 1,048,576
ITERS = 1000
THREADS = 256
BLOCKS = (N + THREADS - 1) // THREADS
TRIALS = 10
# float32 constants: a bare 1.0001 literal would promote the math to float64 (1/32 rate on T4)
MUL = np.float32(1.0001)
ADD = np.float32(0.0001)


@cuda.jit(device=True)
def path1(y):
    # multiply-accumulate
    for _ in range(ITERS):
        y = y * MUL + ADD
    return y


@cuda.jit(device=True)
def path2(y):
    # subtract-divide
    for _ in range(ITERS):
        y = (y - ADD) / MUL
    return y


@cuda.jit
def kernel_a_uniform(d_y):
    idx = cuda.grid(1)
    if idx < d_y.shape[0]:
        d_y[idx] = path1(d_y[idx])


@cuda.jit
def kernel_b_interleaved(d_y):
    idx = cuda.grid(1)
    if idx < d_y.shape[0]:
        if idx % 2 == 0:
            d_y[idx] = path1(d_y[idx])
        else:
            d_y[idx] = path2(d_y[idx])


@cuda.jit
def kernel_c_warp_aligned(d_y):
    idx = cuda.grid(1)
    if idx < d_y.shape[0]:
        warp_id = idx // 32
        if warp_id % 2 == 0:
            d_y[idx] = path1(d_y[idx])
        else:
            d_y[idx] = path2(d_y[idx])


def bench(kernel, d_y):
    kernel[BLOCKS, THREADS](d_y)          # warm-up (JIT compile + first launch)
    cuda.synchronize()
    times = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        kernel[BLOCKS, THREADS](d_y)
        cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def main():
    dev = cuda.get_current_device()
    name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
    print(f"GPU: {name} | Compute Capability: {dev.compute_capability[0]}.{dev.compute_capability[1]}")
    print(f"N = {N:,} float32, {ITERS} iterations/element, grid = {BLOCKS} x {THREADS}\n")

    h_y = np.random.default_rng(230103252).random(N, dtype=np.float32)
    d_y = cuda.to_device(h_y)

    results = [
        ("A: Uniform Path", *bench(kernel_a_uniform, d_y)),
        ("B: Interleaved Divergence (idx % 2)", *bench(kernel_b_interleaved, d_y)),
        ("C: Warp-Aligned Branching (warp_id % 2)", *bench(kernel_c_warp_aligned, d_y)),
    ]
    base = results[0][1]
    print(f"| {'Kernel':<40} | {'Mean time (ms)':>14} | {'Std (ms)':>8} | {'vs A':>6} |")
    print(f"|{'-' * 42}|{'-' * 16}|{'-' * 10}|{'-' * 8}|")
    for label, mean, std in results:
        print(f"| {label:<40} | {mean:>14.3f} | {std:>8.3f} | {mean / base:>5.2f}x |")
    b, c = results[1][1], results[2][1]
    print(f"\nDivergence penalty B vs C (same work, different branch layout): {b / c:.2f}x")


if __name__ == "__main__":
    main()
