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

# The file has been adapted from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import os
import shutil
import subprocess

import setuptools
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

current_dir = os.path.dirname(os.path.realpath(__file__))
jit_include_dirs = ("deep_gemm/include/deep_gemm",)
third_party_include_dirs = (
    "../../../csrc/third_party/cutlass/include/cute",
    "../../../csrc/third_party/cutlass/include/cutlass",
)


class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        self.make_jit_include_symlinks()

    @staticmethod
    def make_jit_include_symlinks():
        # Make symbolic links of third-party include directories
        for d in third_party_include_dirs:
            dirname = d.split("/")[-1]
            src_dir = f"{current_dir}/{d}"
            dst_dir = f"{current_dir}/deep_gemm/include/{dirname}"
            assert os.path.exists(src_dir)
            if os.path.exists(dst_dir):
                assert os.path.islink(dst_dir)
                os.unlink(dst_dir)
            os.symlink(src_dir, dst_dir, target_is_directory=True)


class CustomBuildPy(build_py):
    def run(self):
        # First, prepare the include directories
        self.prepare_includes()

        # Then run the regular build
        build_py.run(self)

    def prepare_includes(self):
        # Create temporary build directory instead of modifying package directory
        build_include_dir = os.path.join(self.build_lib, "deep_gemm/include")
        os.makedirs(build_include_dir, exist_ok=True)

        # Copy third-party includes to the build directory
        for d in third_party_include_dirs:
            dirname = d.split("/")[-1]
            src_dir = os.path.join(current_dir, d)
            dst_dir = os.path.join(build_include_dir, dirname)

            # Remove existing directory if it exists
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)

            # Copy the directory
            shutil.copytree(src_dir, dst_dir)


if __name__ == "__main__":
    # noinspection PyBroadException
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        revision = "+" + subprocess.check_output(cmd).decode("ascii").rstrip()
    except:
        revision = ""

    setuptools.setup(
        name="deep_gemm",
        version="1.0.0" + revision,
        packages=["deep_gemm", "deep_gemm/jit", "deep_gemm/jit_kernels"],
        package_data={
            "deep_gemm": [
                "include/deep_gemm/**/*",
                "include/cute/**/*",
                "include/cutlass/**/*",
            ]
        },
        cmdclass={
            "develop": PostDevelopCommand,
            "build_py": CustomBuildPy,
        },
    )
