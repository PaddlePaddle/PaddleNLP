# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# This code is copied fron NVIDIA apex:
# https://github.com/NVIDIA/apex with minor changes.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup

prop = paddle.device.cuda.get_device_properties()
cc = prop.major * 10 + prop.minor
gencode_flags = ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]

setup(
    name="fast_ln",
    ext_modules=CUDAExtension(
        sources=["fast_ln/ln_api.cpp", "fast_ln/ln_bwd_semi_cuda_kernel.cu", "fast_ln/ln_fwd_cuda_kernel.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                "-I./apex/contrib/csrc/layer_norm/",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ]
            + gencode_flags,
        },
    ),
)


setup(
    name="fused_ln",
    ext_modules=CUDAExtension(
        sources=["fused_ln/layer_norm_cuda.cpp", "fused_ln/layer_norm_cuda_kernel.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                "-I./apex/contrib/csrc/layer_norm/",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "-maxrregcount=50",
            ]
            + gencode_flags,
        },
    ),
)
