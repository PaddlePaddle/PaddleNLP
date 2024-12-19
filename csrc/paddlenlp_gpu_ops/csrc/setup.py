# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from site import getsitepackages

import paddle

paddle_includes = []
for site_packages_path in getsitepackages():
    paddle_includes.append(os.path.join(site_packages_path, "paddle", "include"))
    paddle_includes.append(os.path.join(site_packages_path, "paddle", "include", "third_party"))
    paddle_includes.append(os.path.join(site_packages_path, "nvidia", "cudnn", "include"))


def get_gencode_flags(compiled_all=False):
    if not compiled_all:
        prop = paddle.device.cuda.get_device_properties()
        cc = prop.major * 10 + prop.minor
        return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]
    else:
        return [
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_75,code=sm_75",
            "-gencode",
            "arch=compute_70,code=sm_70",
        ]


def get_sm_version():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return cc


def run_single(func):
    p = multiprocessing.Process(target=func)
    p.start()
    p.join()


def run_multi(func_list):
    processes = []
    for func in func_list:
        processes.append(multiprocessing.Process(target=func))
        processes.append(multiprocessing.Process(target=func))
        processes.append(multiprocessing.Process(target=func))

    for p in processes:
        p.start()

    for p in processes:
        p.join()


cc_flag = get_gencode_flags(compiled_all=False)
cc = get_sm_version()


def setup_fast_ln():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    setup(
        name="fast_ln",
        ext_modules=CUDAExtension(
            include_dirs=paddle_includes,
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
                + cc_flag,
            },
        ),
    )


def setup_fused_ln():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    setup(
        name="fused_ln",
        ext_modules=CUDAExtension(
            include_dirs=paddle_includes,
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
                + cc_flag,
            },
        ),
    )


def setup_causal_conv1d():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    sources = [
        "causal_conv1d/causal_conv1d.cpp",
        "causal_conv1d/causal_conv1d_fwd.cu",
        "causal_conv1d/causal_conv1d_bwd.cu",
        "causal_conv1d/causal_conv1d_update.cu",
    ]

    if cc >= 75:
        cc_flag.append("-DCUDA_BFLOAT16_AVAILABLE")

    extra_compile_args = {
        "cxx": ["-O3"],
        "nvcc": [
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ptxas-options=-v",
            "-lineinfo",
            "--threads",
            "4",
        ]
        + cc_flag,
    }

    setup(
        name="causal_conv1d_cuda_pd",
        ext_modules=CUDAExtension(
            sources=sources,
            extra_compile_args=extra_compile_args,
        ),
    )


def setup_selective_scan():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    real_complex_list = ["real"]
    dtype_list = ["fp16", "fp32"]

    if cc > 75:
        dtype_list.insert(1, "bf16")
        cc_flag.append("-DCUDA_BFLOAT16_AVAILABLE")

    sources = [
        "selective_scan/selective_scan.cpp",
    ]
    for real_or_complex in real_complex_list:
        for dtype in dtype_list:
            sources.append(f"selective_scan/selective_scan_fwd_{dtype}_{real_or_complex}.cu")
            sources.append(f"selective_scan/selective_scan_bwd_{dtype}_{real_or_complex}.cu")

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ptxas-options=-v",
            "-lineinfo",
            "--threads",
            "4",
        ]
        + cc_flag,
    }

    setup(
        name="selective_scan_cuda_pd",
        ext_modules=CUDAExtension(
            include_dirs=paddle_includes,
            sources=sources,
            extra_compile_args=extra_compile_args,
        ),
    )


def setup_paddle_bwd_ops():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    sources = ["paddle_bwd_ops/flash_attn_bwd.cc", "paddle_bwd_ops/add_bwd.cc", "paddle_bwd_ops/matmul_bwd.cc"]
    try:
        from paddle.nn.functional.flash_attention import (  # noqa: F401
            flash_attention_with_sparse_mask,
        )

        sources.append("paddle_bwd_ops/flash_attn_with_sparse_mask_bwd.cc")
    except ImportError:
        from paddle.nn.functional.flash_attention import (  # noqa: F401
            flashmask_attention,
        )

        sources.append("paddle_bwd_ops/flashmask_attn_bwd.cc")

    setup(
        name="paddle_bwd_ops",
        ext_modules=CUDAExtension(
            include_dirs=paddle_includes,
            sources=sources,
        ),
    )


if __name__ == "__main__":
    run_multi(
        [
            setup_fast_ln,
            setup_fused_ln,
            setup_causal_conv1d,
            setup_selective_scan,
            setup_paddle_bwd_ops,
        ],
    )
