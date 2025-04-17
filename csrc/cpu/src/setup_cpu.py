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
import site
import subprocess

from paddle.utils.cpp_extension import CppExtension, setup

# from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# refer: https://note.qidong.name/2018/03/setup-warning-strict-prototypes
# Avoid a gcc warning below:
# cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid
# for C/ObjC but not for C++
class BuildExt(build_ext):
    def build_extensions(self):
        if "-Wstrict-prototypes" in self.compiler.compiler_so:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        super().build_extensions()


def check_avx512_bf16__support():
    try:
        result = subprocess.run(
            ["lscpu", "|", "grep", '"avx512_bf16"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
        )

        if "avx512_bf16" in result.stdout.lower():
            return True
        else:
            return False

    except Exception as e:
        print(f"Error checking AVX512 support: {e}")
        return False


paddle_extra_compile_args = [
    "-std=c++17",
    "-shared",
    "-fPIC",
    "-Wno-parentheses",
    "-DPADDLE_WITH_CUSTOM_KERNEL",
    "-mavx512f",
    "-mavx512vl",
    "-fopenmp",
    "-mavx512bw",
    "-mno-mmx",
    "-Wall",
    "-march=skylake-avx512",
    "-O3",
    "-g",
]

if check_avx512_bf16__support():
    paddle_extra_compile_args += [
        "-DAVX512_BF16_WEIGHT_ONLY_BF16=true",
        "-DAVX512_FP16_WEIGHT_ONLY_INT8=true",
        "-DAVX512_FP16_WEIGHT_ONLY_FP16=true",
    ]
else:
    paddle_extra_compile_args += [
        "-DAVX512_FP32_WEIGHT_ONLY_FP16=true",
        "-DAVX512_FP32_WEIGHT_ONLY_INT8=true",
    ]
# include path
site_packages_path = site.getsitepackages()
paddle_custom_kernel_include = [os.path.join(path, "paddle", "include") for path in site_packages_path]

XFT_INCLUDE_DIR = os.environ["XFT_HEADER_DIR"]
XFT_LIBRARY_DIR = os.environ["XFT_LIB_DIR"]

# include path third_party
paddle_custom_kernel_include += [
    os.path.join(XFT_INCLUDE_DIR, "include"),
    os.path.join(XFT_INCLUDE_DIR, "src/common"),
    os.path.join(XFT_INCLUDE_DIR, "src/kernel"),
    os.path.join(XFT_INCLUDE_DIR, "src/layers"),
    os.path.join(XFT_INCLUDE_DIR, "src/models"),
    os.path.join(XFT_INCLUDE_DIR, "src/utils"),
    os.path.join(XFT_INCLUDE_DIR, "3rdparty/onednn/include"),
    os.path.join(XFT_INCLUDE_DIR, "3rdparty/onednn/build/include"),
    os.path.join(XFT_INCLUDE_DIR, "3rdparty/xdnn"),
    os.path.join(XFT_INCLUDE_DIR, "3rdparty"),
    os.path.join(XFT_INCLUDE_DIR, "3rdparty/mkl/include"),
]

# libs path
paddle_custom_kernel_library_dir = [os.path.join(path, "paddle", "base") for path in site_packages_path]
paddle_custom_kernel_library_dir += [XFT_LIBRARY_DIR]


libs = [":libxfastertransformer.so", ":libxft_comm_helper.so"]

custom_kernel_dot_module = CppExtension(
    sources=[
        "../gpu/save_with_output.cc",
        "./src/token_penalty_multi_scores.cc",
        "./src/stop_generation_multi_ends.cc",
        "./src/set_value_by_flags.cc",
        "./src/xft_transformer.cc",
        "./src/avx_weight_only.cc",
        "./src/xft_greedy_search.cc",
    ],
    include_dirs=paddle_custom_kernel_include,
    library_dirs=paddle_custom_kernel_library_dir,
    libraries=libs,
    extra_compile_args=paddle_extra_compile_args,
)

setup(
    name="paddlenlp_ops",
    version="1.0",
    description="custom kernel for compiling",
    ext_modules=[custom_kernel_dot_module],
)
