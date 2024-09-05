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

import os
import subprocess
from typing import List

import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup


def clone_git_repo(version, repo_url, destination_path) -> bool:
    try:
        subprocess.run(["git", "clone", "-b", version, "--single-branch", repo_url, destination_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git clone {repo_url} operation failed with the following error: {e}")
        print("Please check your network connection or access rights to the repository.")
        print(
            "If the problem persists, please refer to the README file for instructions on how to manually download and install the necessary components."
        )
        return False


def get_sm_version() -> List[int]:
    cc_s = paddle.version.cuda_archs()
    return [int(cc) for cc in cc_s.split()]


def get_gencode_flags() -> List[str]:
    cc_s = get_sm_version()
    flags = []
    for cc in cc_s:
        flags += ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]
    return flags


gencode_flags = get_gencode_flags()
library_path = os.environ.get("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")


sources = [
    "./gpu/save_with_output.cc",
    "./gpu/set_value_by_flags.cu",
    "./gpu/token_penalty_multi_scores.cu",
    "./gpu/token_penalty_multi_scores_v2.cu",
    "./gpu/stop_generation_multi_ends.cu",
    "./gpu/fused_get_rope.cu",
    "./gpu/get_padding_offset.cu",
    "./gpu/qkv_transpose_split.cu",
    "./gpu/rebuild_padding.cu",
    "./gpu/transpose_removing_padding.cu",
    "./gpu/write_cache_kv.cu",
    "./gpu/encode_rotary_qk.cu",
    "./gpu/get_padding_offset_v2.cu",
    "./gpu/rebuild_padding_v2.cu",
    "./gpu/set_value_by_flags_v2.cu",
    "./gpu/stop_generation_multi_ends_v2.cu",
    "./gpu/update_inputs.cu",
    "./gpu/get_output.cc",
    "./gpu/save_with_output_msg.cc",
    "./gpu/write_int8_cache_kv.cu",
    "./gpu/step.cu",
    "./gpu/quant_int8.cu",
    "./gpu/dequant_int8.cu",
    "./gpu/flash_attn_bwd.cc",
    "./gpu/tune_cublaslt_gemm.cu",
]

cutlass_dir = "gpu/cutlass_kernels/cutlass"
nvcc_compile_args = gencode_flags

if not os.path.exists(cutlass_dir) or not os.listdir(cutlass_dir):
    if not os.path.exists(cutlass_dir):
        os.makedirs(cutlass_dir)
    clone_git_repo("v3.5.0", "https://github.com/NVIDIA/cutlass.git", cutlass_dir)

nvcc_compile_args += [
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "-Igpu/cutlass_kernels",
    "-Igpu/cutlass_kernels/cutlass/include",
    "-Igpu/fp8_gemm_with_cutlass",
    "-Igpu",
]

cc = get_sm_version()
if max(cc) >= 80:
    sources += ["gpu/int8_gemm_with_cutlass/gemm_dequant.cu"]

if max(cc) >= 89:
    sources += [
        "gpu/fp8_gemm_with_cutlass/fp8_fp8_half_gemm.cu",
        "gpu/cutlass_kernels/fp8_gemm_fused/fp8_fp8_gemm_scale_bias_act.cu",
        "gpu/fp8_gemm_with_cutlass/fp8_fp8_fp8_dual_gemm.cu",
        "gpu/cutlass_kernels/fp8_gemm_fused/fp8_fp8_dual_gemm_scale_bias_act.cu",
    ]

setup(
    name="paddlenlp_ops",
    ext_modules=CUDAExtension(
        sources=sources,
        extra_compile_args={"cxx": ["-O3"], "nvcc": nvcc_compile_args},
        libraries=["cublasLt"],
        library_dirs=[library_path],
    ),
)
