# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.
"""
this code is modified from https://github.com/state-spaces/mamba/tree/main/mamba_ssm/ops/triton
"""
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

from ...utils import custom_bwd, custom_fwd


def rms_norm_ref(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True, epsilon=None):
    if epsilon is not None:
        eps = epsilon
    dtype = x.dtype
    # N = x.shape[-1]
    weight = weight.cast("float32")
    bias = bias.cast("float32") if bias is not None else None
    if upcast:
        x = x.cast("float32")
        z = z.cast("float32") if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / paddle.sqrt((x.square()).mean(axis=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / paddle.sqrt((x_group.square()).mean(axis=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.cast(dtype)


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        Z += row * stride_z_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        x *= z * tl.sigmoid(z)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        y *= z * tl.sigmoid(z)
    # Write output
    tl.store(Y + cols, y, mask=mask)


def _layer_norm_fwd(x, weight, bias, eps, z=None, out=None, group_size=None, norm_before_gate=True, is_rms_norm=False):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.strides[-1] == 1
    if z is not None:
        assert z.strides[-1] == 1
        assert tuple(z.shape) == (M, N)
    assert weight.shape[0] == N
    assert weight.strides[-1] == 1
    if bias is not None:
        assert bias.strides[-1] == 1
        assert bias.shape[0] == N
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = paddle.empty_like(x)
    assert out.strides[-1] == 1
    mean = paddle.empty((ngroups * M,), dtype=paddle.float32) if not is_rms_norm else None
    rstd = paddle.empty((ngroups * M,), dtype=paddle.float32)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    _layer_norm_fwd_1pass_kernel[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.strides[0],
        out.strides[0],
        z.strides[0] if z is not None else 0,
        M,
        group_size,
        eps,
        BLOCK_N=BLOCK_N,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        num_warps=num_warps,
    )
    return out, mean, rstd


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Y,  # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    DZ,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_z_row,
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dz_row,
    stride_dw_row,
    stride_db_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    group = tl.program_id(1)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row + group * N
    if HAS_Z:
        Z += row_start * stride_z_row + group * N
        DZ += row_start * stride_dz_row + group * N
    DY += row_start * stride_dy_row + group * N
    DX += row_start * stride_dx_row + group * N
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if (RECOMPUTE_OUTPUT or HAS_Z) and HAS_BIAS:
        B += group * N
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z + cols, mask=mask, other=0.0).to(tl.float32)
            x_og = x
            x = x_og * z * tl.sigmoid(z)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z + cols, mask=mask, other=0.0).to(tl.float32)
            z_sigmoid = tl.sigmoid(z)
            y = xhat * w + b if HAS_BIAS else xhat * w
            if RECOMPUTE_OUTPUT:
                tl.store(Y + cols, y * z * z_sigmoid, mask=mask)
            dz = dy * y * z_sigmoid * (1 + z * (1 - z_sigmoid))
            tl.store(DZ + cols, dz, mask=mask)
            dy *= z * z_sigmoid
        else:
            if RECOMPUTE_OUTPUT:
                y = xhat * w + b if HAS_BIAS else xhat * w
                tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        c1 = tl.sum(xhat * wdy, axis=0) / N
        if not IS_RMS_NORM:
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            dx = (wdy - xhat * c1) * rstd
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if HAS_Z and not NORM_BEFORE_GATE:
            z_sigmoid = tl.sigmoid(z)
            dz = dx * x_og * z_sigmoid * (1 + z * (1 - z_sigmoid))
            tl.store(DZ + cols, dz, mask=mask)
            dx *= z * z_sigmoid
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        if HAS_Z:
            Z += stride_z_row
            DZ += stride_dz_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DW + row_block_id * stride_dw_row + group * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * stride_db_row + group * N + cols, db, mask=mask)


def _layer_norm_bwd(
    dy,
    x,
    weight,
    bias,
    eps,
    mean,
    rstd,
    z=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    recompute_output=False,
    dz=None,
    out=None,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.strides[-1] == 1
    assert dy.strides[-1] == 1
    assert tuple(dy.shape) == (M, N)
    if z is not None:
        assert z.strides[-1] == 1
        assert tuple(z.shape) == (M, N)
    assert weight.shape[0] == N
    assert weight.strides[-1] == 1
    if bias is not None:
        assert bias.strides[-1] == 1
        assert bias.shape[0] == N
    # allocate output
    dx = paddle.empty_like(x)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
        assert dz.strides[-1] == 1
    else:
        dz = paddle.empty_like(z) if z is not None else None
    if recompute_output:
        if out is None:
            out = paddle.empty_like(x)
        assert out.shape == x.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    sm_count = paddle.device.cuda.get_device_properties(paddle.get_device()).multi_processor_count
    # If group size is small (e.g., 64), we're only using 1 warp. So having just 108 programs
    # would limit the occupancy.
    nrow_groups = math.ceil(sm_count * math.ceil(4 / num_warps) / ngroups)
    _dw = paddle.empty((nrow_groups, N), dtype=paddle.float32)
    _db = paddle.empty((nrow_groups, N), dtype=paddle.float32) if bias is not None else None
    rows_per_program = math.ceil(M / nrow_groups)
    grid = (nrow_groups, ngroups)
    _layer_norm_bwd_kernel[grid](
        x,
        weight,
        bias,
        z,
        out if recompute_output else None,
        dy,
        dx,
        _dw,
        _db,
        dz,
        mean,
        rstd,
        x.strides[0],
        z.strides[0] if z is not None else 0,
        0 if not recompute_output else out.strides[0],
        dy.strides[0],
        dx.strides[0],
        dz.strides[0] if dz is not None else 0,
        _dw.strides[0],
        _db.strides[0] if _db is not None else 0,
        M,
        group_size,
        eps,
        rows_per_program,
        BLOCK_N=BLOCK_N,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        num_warps=num_warps,
    )
    dw = _dw.sum(0).cast(weight.dtype)
    db = _db.sum(0).cast(bias.dtype) if bias is not None else None
    return (dx, dw, db, dz) if not recompute_output else (dx, dw, db, dz, out)


class LayerNormFn(paddle.autograd.PyLayer):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, is_rms_norm=False):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""

        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape([-1, x.shape[-1]])
        if x.strides[-1] != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape([-1, z.shape[-1]])
            if z.strides[-1] != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = _layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd, z)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.norm_before_gate = norm_before_gate
        ctx.is_rms_norm = is_rms_norm
        return y.reshape(x_shape_og)

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        x, weight, bias, mean, rstd, z = ctx.saved_tensor()
        dy = dy.reshape([-1, dy.shape[-1]])
        if dy.strides[-1] != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        dx, dw, db, dz = _layer_norm_bwd(
            dy, x, weight, bias, ctx.eps, mean, rstd, z, ctx.group_size, ctx.norm_before_gate, ctx.is_rms_norm
        )
        return (
            dx.reshape(ctx.x_shape_og),
            dw,
            db,
            dz.reshape(ctx.x_shape_og) if dz is not None else None,
            None,
            None,
            None,
            None,
        )


def layernorm_fn(
    x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, is_rms_norm=False, epsilon=None
):
    if epsilon is not None:
        eps = epsilon
    return LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)


def rmsnorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, epsilon=None):
    if epsilon is not None:
        eps = epsilon
    return LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, True)


class LayerNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, epsilon=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """

        super().__init__()
        self.eps = epsilon or eps
        self.weight = self.create_parameter(
            shape=[
                hidden_size,
            ],
            default_initializer=nn.initializer.Constant(value=1.0),
            dtype=paddle.get_default_dtype(),
        )
        self.bias = self.create_parameter(
            shape=[
                hidden_size,
            ],
            default_initializer=nn.initializer.Constant(value=0.0),
            dtype=paddle.get_default_dtype(),
        )
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""
        return layernorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            group_size=self.group_size,
            eps=self.eps,
            norm_before_gate=self.norm_before_gate,
        )


class RMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, epsilon=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        super().__init__()
        self.eps = epsilon or eps
        self.weight = self.create_parameter(
            shape=[
                hidden_size,
            ],
            default_initializer=nn.initializer.Constant(value=1.0),
            dtype=paddle.get_default_dtype(),
        )
        self.bias = None
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""
        return rmsnorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )
