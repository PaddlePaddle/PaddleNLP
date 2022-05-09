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
from setuptools import setup

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.

if os.name != 'nt':
    package_data = {
        "faster_tokenizers":
        ["core_tokenizers.so", "libicuuc.so.70", "libicudata.so.70"]
    }
else:
    package_data = {
        "faster_tokenizers":
        ["core_tokenizers.pyd", "icuuc.dll", "icuucdata.dll"]
    }

long_description = "PaddleNLP Faster Tokenizer Library written in C++ "
setup(
    name="faster_tokenizers",
    version="0.0.1",
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
    license='Apache 2.0', )
