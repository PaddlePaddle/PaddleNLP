# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import io
import os

import setuptools

description = "Paddle-Pipelines: An End to End Natural Language Proceessing Development Kit Based on PaddleNLP"

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


setuptools.setup(
    name="paddle-pipelines",
    version=read("VERSION"),
    author="PaddlePaddle Speech and Language Team",
    author_email="paddlenlp@baidu.com",
    description=description,
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/PaddlePaddle/PaddleNLP",
    packages=setuptools.find_packages(where=".", exclude=("examples*", "tests*", "docs*", "ui*", "rest_api*")),
    setup_requires=["cython", "numpy"],
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
)
