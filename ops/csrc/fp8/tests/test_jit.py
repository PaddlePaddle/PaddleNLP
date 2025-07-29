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
import os
from typing import Any, Dict

import cuda.bindings.driver as cbd
import paddle
from deep_gemm import jit

# Essential debugging staffs
os.environ["DG_JIT_DEBUG"] = os.getenv("DG_JIT_DEBUG", "1")
os.environ["DG_JIT_DISABLE_CACHE"] = os.getenv("DG_JIT_DISABLE_CACHE", "1")


class VectorAddRuntime(jit.Runtime):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    @staticmethod
    def generate(kwargs: Dict[str, Any]) -> str:
        return f"""
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#endif

#include <cuda_fp8.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void vector_add(T* a, T* b, T* c, uint32_t n) {{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {{
        c[i] = a[i] + b[i];
    }}
}}

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&vector_add<{kwargs['T']}>);
}}
"""

    # noinspection PyShadowingNames,PyMethodOverriding
    @staticmethod
    def launch(kernel: cbd.CUkernel, kwargs: Dict[str, Any]) -> cbd.CUresult:
        assert kwargs["A"].shape == kwargs["B"].shape == kwargs["C"].shape
        assert kwargs["A"].place == kwargs["B"].place == kwargs["C"].place
        assert kwargs["A"].dim() == 1

        config = cbd.CUlaunchConfig()
        config.gridDimX = (kwargs["A"].numel() + 127) // 128
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 128
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = kwargs["STREAM"]

        arg_values = (
            kwargs["A"].data_ptr(),
            kwargs["B"].data_ptr(),
            kwargs["C"].data_ptr(),
            kwargs["A"].numel(),
        )
        arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )

        return cbd.cuLaunchKernelEx(config, kernel, (arg_values, arg_types), 0)[0]


if __name__ == "__main__":
    print("Generated code:")
    kwargs = {"T": "float"}
    code = VectorAddRuntime.generate(kwargs)
    print(code)
    print()

    for compiler_name in ("NVCC", "NVRTC"):
        # Get compiler
        compiler_cls = getattr(jit, f"{compiler_name}Compiler")
        print(f"Compiler: {compiler_name}, version: {compiler_cls.__version__()}")

        # Build
        print("Building ...")
        func = compiler_cls.build("test_func", code, VectorAddRuntime, kwargs)

        # Run and check
        a = paddle.randn((1024,), dtype=paddle.float32)
        b = paddle.randn((1024,), dtype=paddle.float32)
        c = paddle.empty_like(a)
        ret = func(A=a, B=b, C=c, STREAM=paddle.device.cuda.current_stream().cuda_stream)
        assert ret == cbd.CUresult.CUDA_SUCCESS, ret
        assert paddle.allclose(c, a + b).item()
        print(f"JIT test for {compiler_name} passed\n")
