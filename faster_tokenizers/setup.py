# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import subprocess
import sys
import multiprocessing

import setuptools
from setuptools import setup, Distribution, Extension
from setuptools.command.install import install


class BinaryDistribution(Distribution):
    # when build the package, it will add
    # platform name such as "cp37-cp37m-linux_x86_64"
    def has_ext_modules(self):
        return True


class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


if os.name != 'nt':
    package_data = {
        "faster_tokenizers": [
            "core_tokenizers.so", "libicuuc.so.70", "libicudata.so.70",
            "commit.log"
        ]
    }
else:
    package_data = {
        "faster_tokenizers":
        ["core_tokenizers.pyd", "icuuc.dll", "icuucdata.dll", "commit.log"]
    }


def get_version():
    f = open(os.path.join("python", "faster_tokenizers", "__init__.py"))
    lines = f.readlines()
    version = ""
    for line in lines:
        if line.startswith("__version__"):
            version = line.split("=")[1]
            version = version.strip().replace("\"", "")
            break
    return version


long_description = "PaddleNLP Faster Tokenizer Library written in C++ "
setup(
    name="faster_tokenizers",
    version=get_version(),
    author="PaddlePaddle Speech and Language Team",
    author_email="paddlesl@baidu.com",
    description=long_description,
    long_description=long_description,
    zip_safe=False,
    url="https://github.com/PaddlePaddle/PaddleNLP/faster_tokenizers",
    package_dir={"": "python"},
    packages=[
        "faster_tokenizers", "faster_tokenizers.tokenizers_impl",
        "faster_tokenizers.normalizers", "faster_tokenizers.pretokenizers",
        "faster_tokenizers.models", "faster_tokenizers.postprocessors"
    ],
    package_data=package_data,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.6",
    cmdclass={'install': InstallPlatlib},
    license='Apache 2.0',
    distclass=BinaryDistribution,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ], )
