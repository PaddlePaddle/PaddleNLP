# Copyright (c) 2024, Tri Dao, Albert Gu.
"""
this code is modified from https://github.com/state-spaces/mamba/tree/main/mamba_ssm/ops/triton
"""
import paddle
import triton
import triton.language as tl


@triton.paddle_autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}),
        triton.Config({"BLOCK_N": 64}),
        triton.Config({"BLOCK_N": 128}),
        triton.Config({"BLOCK_N": 256}),
        triton.Config({"BLOCK_N": 512}),
        triton.Config({"BLOCK_N": 1024}),
    ],
    key=["ncols"],
)
@triton.jit
def _swiglu_fwd_kernel(
    X,
    Y,
    OUT,
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_out_row,
    ncols,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += row * stride_x_row
    Y += row * stride_y_row
    OUT += row * stride_out_row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.0).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.0).to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=cols < ncols)


def _swiglu_fwd(xy, out=None):
    if xy.strides[-1] != 1:
        xy = xy.contiguous()
    batch_shape = xy.shape[:-1]
    xy = xy.reshape([-1, xy.shape[-1]])
    x, y = xy.chunk(2, axis=-1)
    if out is None:
        out = paddle.empty_like(x)
    else:
        out = out.reshape([-1, out.shape[-1]])
        assert out.shape == x.shape
    assert out.strides[-1] == 1
    M, N = x.shape
    grid = lambda META: (M, triton.cdiv(N, META["BLOCK_N"]))
    _swiglu_fwd_kernel[grid](x, y, out, x.strides[0], y.strides[0], out.strides[0], N)
    return out.reshape([*batch_shape, out.shape[-1]])


@triton.paddle_autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}),
        triton.Config({"BLOCK_N": 64}),
        triton.Config({"BLOCK_N": 128}),
        triton.Config({"BLOCK_N": 256}),
        triton.Config({"BLOCK_N": 512}),
        triton.Config({"BLOCK_N": 1024}),
    ],
    key=["ncols"],
)
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["OUT"] is not None})
@triton.jit
def _swiglu_bwd_kernel(
    X,
    Y,
    DOUT,
    OUT,
    DX,
    DY,
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_dout_row,
    stride_out_row,
    stride_dx_row,
    stride_dy_row,
    ncols,
    BLOCK_N: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += row * stride_x_row
    Y += row * stride_y_row
    DOUT += row * stride_dout_row
    if RECOMPUTE_OUTPUT:
        OUT += row * stride_out_row
    DX += row * stride_dx_row
    DY += row * stride_dy_row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.0).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.0).to(tl.float32)
    dout = tl.load(DOUT + cols, mask=cols < ncols, other=0.0).to(tl.float32)
    x_sigmoid = tl.sigmoid(x)
    dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
    dy = x * x_sigmoid * dout
    tl.store(DX + cols, dx, mask=cols < ncols)
    tl.store(DY + cols, dy, mask=cols < ncols)
    if RECOMPUTE_OUTPUT:
        out = x * x_sigmoid * y
        tl.store(OUT + cols, out, mask=cols < ncols)


def _swiglu_bwd(xy, dout, dxy=None, recompute_output=False, out=None):
    if xy.strides[-1] != 1:
        xy = xy.contiguous()
    if dout.strides[-1] != 1:
        dout = dout.contiguous()
    batch_shape = xy.shape[:-1]
    xy = xy.reshape([-1, xy.shape[-1]])
    x, y = xy.chunk(2, axis=-1)
    dout = dout.reshape([-1, dout.shape[-1]])
    assert dout.shape == x.shape
    if dxy is None:
        dxy = paddle.empty_like(xy)
    else:
        dxy = dxy.reshape([-1, dxy.shape[-1]])
        assert dxy.shape == xy.shape
    dx, dy = dxy.chunk(2, axis=-1)
    assert dx.strides[-1] == 1
    assert dy.strides[-1] == 1
    if recompute_output:
        if out is None:
            out = paddle.empty_like(x)
        else:
            out = out.reshape([-1, out.shape[-1]])
            assert out.shape == x.shape
        assert out.strides[-1] == 1
    M, N = x.shape
    grid = lambda META: (M, triton.cdiv(N, META["BLOCK_N"]))
    _swiglu_bwd_kernel[grid](
        x,
        y,
        dout,
        out if recompute_output else None,
        dx,
        dy,
        x.strides[0],
        y.strides[0],
        dout.strides[0],
        out.strides[0] if recompute_output else 0,
        dx.strides[0],
        dy.strides[0],
        N,
    )
    if not recompute_output:
        return dxy.reshape([*batch_shape, dxy.shape[-1]])
    else:
        return dxy.reshape([*batch_shape, dxy.shape[-1]]), out.reshape([*batch_shape, out.shape[-1]])


class SwiGLU(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, xy):
        ctx.save_for_backward(xy)
        return _swiglu_fwd(xy)

    @staticmethod
    def backward(ctx, dout):
        (xy,) = ctx.saved_tensor()
        return _swiglu_bwd(xy, dout)


swiglu = SwiGLU.apply
