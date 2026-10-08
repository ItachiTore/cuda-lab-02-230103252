"""Task 3: arbitrary-size vector scaling with a grid-stride loop.

N = 2^24 elements are processed by only 64 x 256 = 16,384 hardware threads:
each thread walks the array in steps of the whole grid size (1,024 elements per thread).
"""
import time

import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = 64


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    h_arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    d_arr = cuda.to_device(h_arr)
    grid_stride_scale_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](d_arr, np.float32(factor), h_arr.shape[0])
    cuda.synchronize()
    return d_arr.copy_to_host()


def main():
    N = 1 << 24   # 16,777,216
    factor = 4.25
    total_threads = BLOCKS_PER_GRID * THREADS_PER_BLOCK
    h_arr = np.ones(N, dtype=np.float32)

    run_grid_stride(h_arr[:1024], factor)     # warm-up (JIT compile)
    t0 = time.perf_counter()
    res = run_grid_stride(h_arr, factor)
    elapsed = (time.perf_counter() - t0) * 1000.0

    mismatches = int(np.count_nonzero(res != np.float32(factor)))
    assert res.shape[0] == N and mismatches == 0, f"{mismatches} elements not scaled"
    print(f"N = {N:,} elements, launched {BLOCKS_PER_GRID} x {THREADS_PER_BLOCK} = {total_threads:,} threads "
          f"-> {N // total_threads:,} elements per thread")
    print(f"Roundtrip time (H2D + kernel + D2H): {elapsed:.3f} ms")
    print(f"TASK 3 PASSED: all {N:,} elements == {factor} (mismatches = {mismatches})")


if __name__ == "__main__":
    main()
