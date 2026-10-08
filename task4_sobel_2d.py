"""Task 4: 2D Sobel horizontal (K_x) filter on a 2048 x 2048 float32 matrix.

K_x = [[-1, 0, +1],
       [-2, 0, +2],
       [-1, 0, +1]]
Interior pixels get the convolution; the one-pixel outer border is set to 0.0.
"""
import numpy as np
from numba import cuda

THREADS_2D = (16, 16)


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)
    if row < rows and col < cols:
        if 0 < row < rows - 1 and 0 < col < cols - 1:
            d_out[row, col] = (-1.0 * d_in[row - 1, col - 1] + 1.0 * d_in[row - 1, col + 1]
                               - 2.0 * d_in[row, col - 1] + 2.0 * d_in[row, col + 1]
                               - 1.0 * d_in[row + 1, col - 1] + 1.0 * d_in[row + 1, col + 1])
        else:
            d_out[row, col] = 0.0


def grid_for(rows, cols):
    # x covers columns, y covers rows (cuda.grid(2) returns (x, y) = (col, row))
    return ((cols + THREADS_2D[0] - 1) // THREADS_2D[0],
            (rows + THREADS_2D[1] - 1) // THREADS_2D[1])


def run_sobel(h_img):
    h_img = np.ascontiguousarray(h_img, dtype=np.float32)
    rows, cols = h_img.shape
    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array_like(d_in)
    sobel_x_kernel[grid_for(rows, cols), THREADS_2D](d_in, d_out, rows, cols)
    cuda.synchronize()
    return d_out.copy_to_host()


def cpu_sobel_x(img):
    out = np.zeros_like(img)
    out[1:-1, 1:-1] = (-img[:-2, :-2] + img[:-2, 2:]
                       - 2.0 * img[1:-1, :-2] + 2.0 * img[1:-1, 2:]
                       - img[2:, :-2] + img[2:, 2:])
    return out


def main():
    rows = cols = 2048
    h_img = np.random.default_rng(230103252).random((rows, cols), dtype=np.float32)
    res = run_sobel(h_img)
    ref = cpu_sobel_x(h_img)
    bx, by = grid_for(rows, cols)
    delta = float(np.max(np.abs(res - ref)))
    assert np.allclose(res, ref, atol=1e-4), f"max delta {delta}"
    assert not res[0, :].any() and not res[-1, :].any() and not res[:, 0].any() and not res[:, -1].any()

    # sanity check on a horizontal ramp: d/dx of x is constant -> interior = 8 * slope
    ramp = np.tile(np.arange(64, dtype=np.float32), (64, 1))
    ramp_out = run_sobel(ramp)
    print(f"Input {rows} x {cols}, block {THREADS_2D[0]} x {THREADS_2D[1]}, "
          f"grid {bx} x {by} = {bx * by:,} blocks ({bx * by * 256:,} threads)")
    print(f"Max |GPU - CPU| = {delta:.3e}; border rows/cols all zero")
    print(f"Ramp test (x-gradient = 1): interior value = {ramp_out[32, 32]:.1f} (expected 8.0)")
    print("TASK 4 PASSED")


if __name__ == "__main__":
    main()
