"""Task 2: 3-point smoothing stencil with boundary clamping (halo replication).

y[i] = 0.25 * x[i-1] + 0.5 * x[i] + 0.25 * x[i+1], with x[-1] := x[0] and x[N] := x[N-1].
"""
import numpy as np
from numba import cuda

THREADS = 256


@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)
    if idx < N:
        # clamp the neighbours at both ends instead of reading outside the buffer
        left = d_in[0] if idx == 0 else d_in[idx - 1]
        right = d_in[N - 1] if idx == N - 1 else d_in[idx + 1]
        d_out[idx] = 0.25 * left + 0.5 * d_in[idx] + 0.25 * right


def run_stencil(h_in):
    h_in = np.ascontiguousarray(h_in, dtype=np.float32)
    N = h_in.shape[0]
    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(d_in)
    blocks = (N + THREADS - 1) // THREADS
    stencil_1d[blocks, THREADS](d_in, d_out, N)
    cuda.synchronize()
    return d_out.copy_to_host()


def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode='edge')
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


def main():
    N = 100_007   # odd, non-power-of-two
    h_in = np.random.default_rng(230103252).standard_normal(N).astype(np.float32)
    h_out_gpu = run_stencil(h_in)
    cpu_ref = cpu_stencil(h_in)
    assert np.allclose(h_out_gpu, cpu_ref, atol=1e-4)
    delta = float(np.max(np.abs(h_out_gpu - cpu_ref)))
    print(f"N = {N:,}, blocks = {(N + THREADS - 1) // THREADS} x {THREADS} threads "
          f"({(N + THREADS - 1) // THREADS * THREADS - N} idle guarded threads)")
    print(f"Edges: out[0] = {h_out_gpu[0]:.6f} (cpu {cpu_ref[0]:.6f}), "
          f"out[N-1] = {h_out_gpu[-1]:.6f} (cpu {cpu_ref[-1]:.6f})")
    print(f"TASK 2 PASSED: MAX DELTA = {delta:.3e}")


if __name__ == "__main__":
    main()
