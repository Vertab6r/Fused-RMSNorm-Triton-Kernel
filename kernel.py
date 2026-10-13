# fused rmsnorm in triton: one gpu program per row. it reads the row once,
# does all the math on-chip, and writes the result once

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, row_stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols  # BLOCK is rounded up to a power of 2, so mask the extra slots

    # other=0.0 is fine for masked slots, zeros don't change a sum of squares
    # fp32 because squaring and summing fp16 can overflow / lose precision
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / n_cols  # real length, not BLOCK
    rstd = tl.rsqrt(mean_sq + eps)

    y = x * rstd * w
    tl.store(y_ptr + row * row_stride + cols, y, mask=mask)


def rmsnorm(x, weight, eps=1e-6):
    assert x.is_cuda and x.shape[-1] == weight.shape[0]

    orig_shape = x.shape
    # flatten (batch, seq, hidden) -> (rows, hidden). contiguous because the
    # kernel assumes rows are laid out back to back
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    n_rows, n_cols = x2d.shape
    y = torch.empty_like(x2d)

    BLOCK = triton.next_power_of_2(n_cols)

    # more warps for bigger rows. these cutoffs are a guess, not tuned
    if BLOCK <= 2048:
        num_warps = 4
    elif BLOCK <= 8192:
        num_warps = 8
    else:
        num_warps = 16

    rmsnorm_kernel[(n_rows,)](
        x2d, weight, y, x2d.stride(0), n_cols, eps,
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return y.reshape(orig_shape)
