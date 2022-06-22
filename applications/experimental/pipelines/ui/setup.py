from setuptools import setup, find_packages
import logging
from pathlib import Path

VERSION = "0.0.1a0"

setup(
    name="pipelines-ui",
    version=VERSION,
    description=
    "Demo UI for pipelines (https://github.com/PaddlePaddle/pipelines)",
    author="paddlenlp",
    author_email="paddlenlp@baidu.com",
    url=" https://github.com/PaddlePaddle/pipelines/tree/master/ui",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(),
    python_requires=">=3.7, <4",
    install_requires=[
        "streamlit>=1.2.0, <=1.7.0", "st-annotated-text>=2.0.0, <3",
        "markdown>=3.3.4, <4"
    ],
)
