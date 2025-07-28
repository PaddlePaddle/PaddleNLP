# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The file has been adapted from DeepSeek DeepGEMM project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepGEMM/blob/main/LICENSE

import ctypes
import enum
import os
from typing import Any, Dict, Tuple

import cuda.bindings.driver as cbd
import paddle

from ..jit.runtime import Runtime
from .utils import get_tma_aligned_size


class GemmType(enum.Enum):
    Normal = 0
    GroupedContiguous = 1
    GroupedMasked = 2

    def __str__(self) -> str:
        return {
            0: "Normal",
            1: "GroupedContiguous",
            2: "GroupedMasked",
        }[self.value]


# TODO Support dtype in Paddle
tmap_type_map: Dict[Any, str] = {
    paddle.int8: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    paddle.int16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    paddle.int32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
    paddle.int64: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT64,
    paddle.uint8: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    # paddle.uint16:          cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    # paddle.uint32:          cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32,
    # paddle.uint64:          cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64,
    paddle.float32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    paddle.float16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    paddle.bfloat16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    paddle.float8_e4m3fn: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    # paddle.float8_e4m3fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    paddle.float8_e5m2: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    # paddle.float8_e5m2fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
}

swizzle_type_map = {
    0: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    32: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}


def get_num_math_warpgroups(block_m: int) -> int:
    return 1 if block_m == 64 else 2


def get_num_threads_per_sm(num_tma_threads: int, num_math_threads_per_group: int, block_m: int) -> int:
    assert num_math_threads_per_group == 128, "Only support 128 threads per math group"
    return get_num_math_warpgroups(block_m) * num_math_threads_per_group + num_tma_threads


def make_2d_tma_copy_desc(
    t: paddle.Tensor,
    gmem_dims: Tuple[cbd.cuuint64_t, cbd.cuuint64_t],
    gmem_outer_stride: cbd.cuuint64_t,
    smem_dims: Tuple[cbd.cuuint32_t, cbd.cuuint32_t],
    swizzle_type: cbd.CUtensorMapSwizzle,
) -> cbd.CUtensorMap:
    tensor_dtype = tmap_type_map[t.dtype]
    res, tensor_map = cbd.cuTensorMapEncodeTiled(
        tensor_dtype,
        2,
        t.data_ptr(),
        gmem_dims,
        (gmem_outer_stride,),
        smem_dims,
        (cbd.cuuint32_t(1), cbd.cuuint32_t(1)),
        cbd.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_type,
        cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cbd.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )

    if res != cbd.CUresult.CUDA_SUCCESS:
        raise Exception(f"Failed to encode tensor map: {res}")
    return tensor_map


def make_2d_tma_desc(
    t: paddle.Tensor,
    gmem_inner_dim: int,
    gmem_outer_dim: int,
    gmem_outer_stride: int,
    smem_inner_dim: int,
    smem_outer_dim: int,
    swizzle_type: cbd.CUtensorMapSwizzle = cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
) -> cbd.CUtensorMap:
    gmem_dim = (cbd.cuuint64_t(gmem_inner_dim), cbd.cuuint64_t(gmem_outer_dim))
    smem_dim = (cbd.cuuint32_t(smem_inner_dim), cbd.cuuint32_t(smem_outer_dim))
    return make_2d_tma_copy_desc(
        t, gmem_dim, cbd.cuuint64_t(gmem_outer_stride * t.element_size()), smem_dim, swizzle_type
    )


def make_2d_tma_a_desc(
    gemm_type: GemmType,
    t: paddle.Tensor,
    shape_m: int,
    shape_k: int,
    m_stride: int,
    block_m: int,
    block_k: int,
    num_groups: int,
) -> cbd.CUtensorMap:
    return make_2d_tma_desc(
        t, shape_k, shape_m * (num_groups if gemm_type == GemmType.GroupedMasked else 1), m_stride, block_k, block_m
    )


def make_2d_tma_b_desc(
    gemm_type: GemmType,
    t: paddle.Tensor,
    shape_n: int,
    shape_k: int,
    n_stride: int,
    block_n: int,
    block_k: int,
    num_groups: int,
) -> cbd.CUtensorMap:
    return make_2d_tma_desc(
        t, shape_k, shape_n * (num_groups if gemm_type != GemmType.Normal else 1), n_stride, block_k, block_n
    )


def make_2d_tma_d_desc(
    gemm_type: GemmType,
    t: paddle.Tensor,
    shape_m: int,
    shape_n: int,
    m_stride: int,
    block_m: int,
    block_n: int,
    num_groups: int,
    swizzle_mode: int,
) -> cbd.CUtensorMap:
    # Swizzling requires the inner box dim to be less or equal than `kSwizzleDMode`
    # bytes, so `BLOCK_N * sizeof(T) / kSwizzleDMode` TMA stores are required
    return make_2d_tma_desc(
        t,
        shape_n,
        shape_m * (num_groups if gemm_type == GemmType.GroupedMasked else 1),
        m_stride,
        block_n if swizzle_mode == 0 else swizzle_mode // t.element_size(),
        block_m,
        swizzle_type_map[swizzle_mode],
    )


def make_2d_tma_scales_desc(
    gemm_type: GemmType, t: paddle.Tensor, shape_mn: int, shape_k: int, block_mn: int, block_k: int, num_groups: int
) -> cbd.CUtensorMap:
    # Make TMA aligned to 16 bytes
    shape_mn = get_tma_aligned_size(shape_mn, t.element_size())
    return make_2d_tma_desc(
        t,
        shape_mn,
        (shape_k + block_k - 1) // block_k * (num_groups if gemm_type == GemmType.GroupedMasked else 1),
        shape_mn,
        block_mn,
        1,
        cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    )


class FP8GemmRuntime(Runtime):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    @staticmethod
    def generate(kwargs: Dict[str, Any]) -> str:
        code = f"""
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <deep_gemm/fp8_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&fp8_gemm_kernel<
        {kwargs['N']},
        {kwargs['K']},
        {kwargs['BLOCK_M']},
        {kwargs['BLOCK_N']},
        {kwargs['BLOCK_K']},
        {kwargs['BLOCK_N_PADDING']},
        {kwargs['SWIZZLE_D_MODE']},
        {kwargs['NUM_GROUPS']},
        {kwargs['NUM_STAGES']},
        {kwargs['NUM_TMA_THREADS']},
        {kwargs['NUM_MATH_THREADS_PER_GROUP']},
        {kwargs['NUM_TMA_MULTICAST']},
        {'true' if kwargs['IS_TMA_MULTICAST_ON_A'] else 'false'},
        GemmType::{kwargs['GEMM_TYPE']}
      >);
}};
"""
        if int(os.getenv("DG_JIT_DEBUG", 0)):
            print(f"Generated FP8 GEMM code:\n{code}")
        return code

    # noinspection PyMethodOverriding
    @staticmethod
    def launch(kernel: cbd.CUkernel, kwargs: Dict[str, Any]) -> cbd.CUresult:
        num_tma_threads = 128
        num_math_threads_per_group = 128

        result = cbd.cuKernelSetAttribute(
            cbd.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            kwargs["SMEM_SIZE"],
            kernel,
            cbd.CUdevice(kwargs["DEVICE_INDEX"]),
        )[0]
        assert result == cbd.CUresult.CUDA_SUCCESS, f"Failed to set max dynamic shared memory size: {result}"

        attr_val = cbd.CUlaunchAttributeValue()
        attr_val.clusterDim.x = kwargs["NUM_TMA_MULTICAST"]
        attr_val.clusterDim.y = 1
        attr_val.clusterDim.z = 1
        attr = cbd.CUlaunchAttribute()
        attr.id = cbd.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value = attr_val

        config = cbd.CUlaunchConfig()
        config.numAttrs = 1
        config.attrs = [attr]
        config.gridDimX = kwargs["NUM_SMS"]
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = get_num_threads_per_sm(num_tma_threads, num_math_threads_per_group, kwargs["BLOCK_M"])
        config.blockDimY = 1
        config.blockDimZ = 1
        config.sharedMemBytes = kwargs["SMEM_SIZE"]
        config.hStream = kwargs["STREAM"]

        arg_values = (
            kwargs["SCALES_B"].data_ptr(),
            kwargs["GROUPED_LAYOUT"].data_ptr(),
            kwargs["M"],
            kwargs["TENSOR_MAP_A"],
            kwargs["TENSOR_MAP_B"],
            kwargs["TENSOR_MAP_SCALES_A"],
            kwargs["TENSOR_MAP_D"],
        )
        arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            None,
            None,
            None,
            None,
        )
        return cbd.cuLaunchKernelEx(config, kernel, (arg_values, arg_types), 0)


class FP8WGradGemmRuntime(Runtime):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    @staticmethod
    def generate(kwargs: Dict[str, Any]) -> str:
        code = f"""
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <deep_gemm/fp8_wgrad_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&fp8_wgrad_gemm_kernel<
        {kwargs['M']},
        {kwargs['N']},
        {kwargs['BLOCK_M']},
        {kwargs['BLOCK_N']},
        {kwargs['BLOCK_K']},
        {kwargs['NUM_STAGES']},
        {kwargs['NUM_LAST_STAGES']},
        {kwargs['NUM_TMA_THREADS']},
        {kwargs['NUM_MATH_THREADS_PER_GROUP']},
        {kwargs['NUM_TMA_MULTICAST']},
        {'true' if kwargs['IS_TMA_MULTICAST_ON_A'] else 'false'}
      >);
}};
"""
        if int(os.getenv("DG_JIT_DEBUG", 0)):
            print(f"Generated FP8 WGrad GEMM code:\n{code}")
        return code

    # noinspection PyMethodOverriding
    @staticmethod
    def launch(kernel: cbd.CUkernel, kwargs: Dict[str, Any]) -> cbd.CUresult:
        num_tma_threads = 128
        num_math_threads_per_group = 128

        result = cbd.cuKernelSetAttribute(
            cbd.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            kwargs["SMEM_SIZE"],
            kernel,
            cbd.CUdevice(kwargs["DEVICE_INDEX"]),
        )[0]
        assert result == cbd.CUresult.CUDA_SUCCESS, f"Failed to set max dynamic shared memory size: {result}"

        attr_val = cbd.CUlaunchAttributeValue()
        attr_val.clusterDim.x = kwargs["NUM_TMA_MULTICAST"]
        attr_val.clusterDim.y = 1
        attr_val.clusterDim.z = 1
        attr = cbd.CUlaunchAttribute()
        attr.id = cbd.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value = attr_val

        config = cbd.CUlaunchConfig()
        config.numAttrs = 1
        config.attrs = [attr]
        config.gridDimX = kwargs["NUM_SMS"]
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = get_num_threads_per_sm(num_tma_threads, num_math_threads_per_group, kwargs["BLOCK_M"])
        config.blockDimY = 1
        config.blockDimZ = 1
        config.sharedMemBytes = kwargs["SMEM_SIZE"]
        config.hStream = kwargs["STREAM"]

        arg_values = (
            kwargs["K"],
            kwargs["TENSOR_MAP_A"],
            kwargs["TENSOR_MAP_B"],
            kwargs["TENSOR_MAP_SCALES_A"],
            kwargs["TENSOR_MAP_SCALES_B"],
            kwargs["TENSOR_MAP_D"],
        )
        arg_types = (
            ctypes.c_uint32,
            None,
            None,
            None,
            None,
            None,
        )
        return cbd.cuLaunchKernelEx(config, kernel, (arg_values, arg_types), 0)
