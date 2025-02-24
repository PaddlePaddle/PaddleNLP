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

""" setup for EfficentLLM """

import os

from setuptools import find_packages, setup

description = "Paddlenlp_ops : inference framework implemented based on PaddlePaddle"
VERSION = "0.0.0"


def read(file: str):
    """
    read file and return content
    """
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, file)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content


def read_version():
    """
    read version and return content
    """
    return VERSION


def read_readme():
    """
    read README.md and return content
    """
    return read("README.md")


setup(
    name="paddlenlp_ops",
    packages=find_packages(),
    version="0.0.0",
    author="Paddle Infernce Team",
    author_email="paddle-inference@baidu.com",
    description=description,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="",
    python_requires=">=3.8",
    package_dir={"paddlenlp_ops": "paddlenlp_ops/"},
    package_data={"paddlenlp_ops": ["sm70/*", "sm75/*", "sm80/*", "sm86/*", "sm89/*", "sm90/*"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
)
