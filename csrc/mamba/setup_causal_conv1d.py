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

import os
from pathlib import Path
from site import getsitepackages

from paddle.utils.cpp_extension import CUDAExtension, setup

this_dir = os.path.dirname(os.path.abspath(__file__))


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


paddle_includes = [
    os.path.dirname(os.path.abspath(__file__)),
    str(Path(this_dir) / "csrc" / "causal_conv1d"),
]
for site_packages_path in getsitepackages():
    paddle_includes.append(os.path.join(site_packages_path, "paddle", "include"))
    paddle_includes.append(os.path.join(site_packages_path, "paddle", "include", "third_party"))


import paddle

prop = paddle.device.cuda.get_device_properties()
cc = prop.major * 10 + prop.minor
cc_list = [
    cc,
]
cc_flag = []
for arch in cc_list:
    cc_flag.append("-gencode")
    cc_flag.append(f"arch=compute_{arch},code=sm_{arch}")

sources = [
    "csrc/causal_conv1d/causal_conv1d.cpp",
    "csrc/causal_conv1d/causal_conv1d_fwd.cu",
    "csrc/causal_conv1d/causal_conv1d_bwd.cu",
    "csrc/causal_conv1d/causal_conv1d_update.cu",
]

if cc >= 75:
    cc_flag.append("-DCUDA_BFLOAT16_AVAILABLE")

print("sources", sources)

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": append_nvcc_threads(
        [
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
        ]
        + cc_flag
    ),
}

setup(
    name="causal_conv1d_cuda_paddle",
    ext_modules=CUDAExtension(
        sources=sources,
        include_dirs=paddle_includes,
        extra_compile_args=extra_compile_args,
        verbose=True,
    ),
)
