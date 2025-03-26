# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import multiprocessing
import os


def get_gencode_flags():
    import paddle

    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]

def run(func):
    p = multiprocessing.Process(target=func)
    p.start()
    p.join()


def change_pwd():
    path = os.path.dirname(__file__)
    if path:
        os.chdir(path)


def setup_fast_ln():
    from paddle.utils.cpp_extension import CUDAExtension, setup
    from paddle.device import is_compiled_with_rocm

    if(is_compiled_with_rocm()):
        print("The 'fasl_ln' feature  is temporarily not supported on the ROCm platform !!!")
    else:
        gencode_flags = get_gencode_flags()
        change_pwd()
        setup(
            name="fast_ln",
            ext_modules=CUDAExtension(
                sources=[
                    "fast_ln/ln_api.cpp",
                    "fast_ln/ln_bwd_semi_cuda_kernel.cu",
                    "fast_ln/ln_fwd_cuda_kernel.cu",
                ],
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


def setup_fused_ln():
    from paddle.utils.cpp_extension import CUDAExtension, setup
    from paddle.device import is_compiled_with_rocm

    gencode_flags = get_gencode_flags()
    change_pwd()
    if(is_compiled_with_rocm()):
        setup(
            name="fused_ln",
            ext_modules=CUDAExtension(
                sources=[
                    "fused_ln/layer_norm_cuda.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "hipcc": [
                        "-O3",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                        "-DPADDLE_WITH_HIP",
                    ]
                },
            ),
        )
    else:
        setup(
            name="fused_ln",
            ext_modules=CUDAExtension(
                sources=[
                    "fused_ln/layer_norm_cuda.cu",
                ],
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

def setup_fused_quant_ops():
    """setup_fused_fp8_ops"""
    from paddle.utils.cpp_extension import CUDAExtension, setup

    gencode_flags = get_gencode_flags()
    change_pwd()
    setup(
        name="FusedQuantOps",
        ext_modules=CUDAExtension(
            sources=[
                "fused_quanted_ops/fused_swiglu_act_quant.cu",
                "fused_quanted_ops/fused_act_quant.cu",
                "fused_quanted_ops/fused_act_dequant.cu",
                "fused_quanted_ops/fused_act_dequant_transpose_act_quant.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-w",
                    "-Wno-abi",
                    "-fPIC",
                    "-std=c++17"
                ],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "-DCUTE_ARCH_MMA_SM90A_ENABLE",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "-lineinfo",
                    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
                    "-maxrregcount=50",
                    "-gencode=arch=compute_90a,code=sm_90a",
                    "-DNDEBUG"
                ] + gencode_flags,
            },
        ),
    )

def setup_token_dispatcher_utils():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    change_pwd()
    setup(
        name="TokenDispatcherUtils",
        ext_modules=CUDAExtension(
            sources=[
                "token_dispatcher_utils/topk_to_multihot.cu",
                "token_dispatcher_utils/topk_to_multihot_grad.cu",
                "token_dispatcher_utils/tokens_unzip_and_zip.cu",
                "token_dispatcher_utils/tokens_guided_unzip.cu",
                "token_dispatcher_utils/regroup_tokens.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-w",
                    "-Wno-abi",
                    "-fPIC",
                    "-std=c++17"
                ],
            "nvcc": [
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                "-DCUTE_ARCH_MMA_SM90A_ENABLE",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "-lineinfo",
                "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
                "-gencode=arch=compute_90a,code=sm_90a",
                "-DNDEBUG"
            ]
            },
        ),
    )

run(setup_token_dispatcher_utils)
run(setup_fused_quant_ops)
run(setup_fast_ln)
run(setup_fused_ln)
