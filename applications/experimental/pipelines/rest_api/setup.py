from setuptools import setup, find_packages
import logging
from pathlib import Path

VERSION = "0.0.1a0"

setup(
    name="pipelines-rest-api",
    version=VERSION,
    description="Demo REST API server for pipelines",
    author="paddlenlp",
    author_email="paddlenlp@baidu.com",
    url=" https://github.com/PaddlePaddle/pipelines/tree/master/rest_api",
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
        # The link below cannot be translated properly into setup.cfg
        # because it looks into the parent folder.
        # TODO check if this is still a limitation later on
        f"pipelines @ file://localhost/{Path(__file__).parent.parent}#egg=pipelines",
        "fastapi<1",
        "uvicorn<1",
        "gunicorn<21",
        "python-multipart<1",  # optional FastAPI dependency for form data
    ],
)
