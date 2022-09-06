# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import io
import paddlenlp

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def get_package_data_files(package, data, package_dir=None):
    """
    Helps to list all specified files in package including files in directories
    since `package_data` ignores directories.
    """
    if package_dir is None:
        package_dir = os.path.join(*package.split('.'))
    all_files = []
    for f in data:
        path = os.path.join(package_dir, f)
        if os.path.isfile(path):
            all_files.append(f)
            continue
        for root, _dirs, files in os.walk(path, followlinks=True):
            root = os.path.relpath(root, package_dir)
            for file in files:
                file = os.path.join(root, file)
                if file not in all_files:
                    all_files.append(file)
    return all_files


setuptools.setup(
    name="paddlenlp",
    version=paddlenlp.__version__,
    author="PaddleNLP Team",
    author_email="paddlenlp@baidu.com",
    description=
    "Easy-to-use and powerful NLP library with Awesome model zoo, supporting wide-range of NLP tasks from research to industrial applications, including Neural Search, Question Answering, Information Extraction and Sentiment Analysis end-to-end system.",
    long_description=read("README_en.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/PaddlePaddle/PaddleNLP",
    packages=setuptools.find_packages(
        where='.',
        exclude=('examples*', 'tests*', 'applications*', 'faster_tokenizer*',
                 'faster_generation*', 'model_zoo*')),
    package_data={
        'paddlenlp.ops':
        get_package_data_files('paddlenlp.ops', [
            'CMakeLists.txt', 'README.md', 'cmake', 'faster_transformer',
            'patches', 'optimizer'
        ]),
        'paddlenlp.transformers.layoutxlm':
        get_package_data_files('paddlenlp.transformers.layoutxlm',
                               ['visual_backbone.yaml']),
    },
    setup_requires=['cython', 'numpy'],
    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    license='Apache 2.0')
