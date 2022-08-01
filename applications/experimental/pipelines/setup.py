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
import os
import setuptools
import sys
import pipelines
import platform

long_description = "PIPELINES: An End to End Natural Language Proceessing Development Kit Based on ERNIE"

if platform.system().lower() == 'windows':
    pass
elif platform.system().lower() == "darwin":
    with open("requirements-cpu.txt") as fin:
        REQUIRED_PACKAGES = fin.read()
elif platform.system().lower() == 'linux':
    with open("requirements.txt") as fin:
        REQUIRED_PACKAGES = fin.read()

setuptools.setup(name="pipelines",
                 version=pipelines.__version__,
                 author="PaddlePaddle Speech and Language Team",
                 author_email="paddlenlp@baidu.com",
                 description=long_description,
                 long_description=long_description,
                 long_description_content_type="text/plain",
                 url="https://github.com/PaddlePaddle/PaddleNLP",
                 packages=setuptools.find_packages(
                     where='.',
                     exclude=('examples*', 'tests*', 'docs*', 'ui*',
                              'rest_api*')),
                 setup_requires=['cython', 'numpy'],
                 install_requires=REQUIRED_PACKAGES,
                 python_requires='>=3.7',
                 classifiers=[
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'License :: OSI Approved :: Apache Software License',
                     'Operating System :: OS Independent',
                 ],
                 license='Apache 2.0')
