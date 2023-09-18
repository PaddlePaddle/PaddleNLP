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
import errno
import io
import os
import subprocess

import setuptools

PADDLENLP_STABLE_VERSION = "PADDLENLP_STABLE_VERSION"


def read_requirements_file(filepath):
    with open(filepath) as fin:
        requirements = fin.read()
    return requirements


def is_git_repo(dir: str) -> bool:
    """Is the given directory version-controlled with git?"""
    return os.path.exists(os.path.join(dir, ".git"))


def have_git() -> bool:
    """Can we run the git executable?"""
    try:
        subprocess.check_output(["git", "--help"])
        return True
    except subprocess.CalledProcessError:
        return False
    except OSError:
        return False


def git_revision(dir: str) -> bytes:
    """Get the SHA-1 of the HEAD of a git repository."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dir).strip()


def git_checkout(dir: str, filename: str) -> bytes:
    """Get the SHA-1 of the HEAD of a git repository."""
    return subprocess.check_output(["git", "checkout", filename], cwd=dir).strip()


def is_dirty(dir: str) -> bool:
    """Check whether a git repository has uncommitted changes."""
    output = subprocess.check_output(["git", "status", "-uno", "--porcelain"], cwd=dir)
    return output.strip() != b""


commit = "unknown"
paddlenlp_dir = os.path.abspath(os.path.dirname(__file__))
if commit.endswith("unknown") and is_git_repo(paddlenlp_dir) and have_git():
    commit = git_revision(paddlenlp_dir).decode("utf-8")
    if is_dirty(paddlenlp_dir):
        commit += ".dirty"


def write_version_py(filename="paddlenlp/version/__init__.py"):
    cnt = '''# THIS FILE IS GENERATED FROM PADDLENLP SETUP.PY
commit           = '%(commit)s'

__all__ = ['show']

def show():
    """Get the corresponding commit id of paddlenlp.

    Returns:
        The commit-id of paddlenlp will be output.

        full_version: version of paddlenlp


    Examples:
        .. code-block:: python

            import paddlenlp

            paddlenlp.version.show()
            # commit: 1ef5b94a18773bb0b1bba1651526e5f5fc5b16fa

    """
    print("commit:", commit)

'''
    commit_id = commit
    content = cnt % {"commit": commit_id}

    dirname = os.path.dirname(filename)

    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with open(filename, "w") as f:
        f.write(content)


__version__ = "2.6.1.post"
if os.getenv(PADDLENLP_STABLE_VERSION):
    __version__ = __version__.replace(".post", "")


extras = {}
REQUIRED_PACKAGES = read_requirements_file("requirements.txt")
extras["tests"] = read_requirements_file("tests/requirements.txt")
extras["docs"] = read_requirements_file("docs/requirements.txt")
extras["autonlp"] = read_requirements_file("paddlenlp/experimental/autonlp/requirements.txt")
extras["dev"] = extras["tests"] + extras["docs"] + extras["autonlp"]


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def get_package_data_files(package, data, package_dir=None):
    """
    Helps to list all specified files in package including files in directories
    since `package_data` ignores directories.
    """
    if package_dir is None:
        package_dir = os.path.join(*package.split("."))
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


if commit != "unknown":
    write_version_py(filename="paddlenlp/version/__init__.py")

try:
    setuptools.setup(
        name="paddlenlp",
        version=__version__,
        author="PaddleNLP Team",
        author_email="paddlenlp@baidu.com",
        description="Easy-to-use and powerful NLP library with Awesome model zoo, supporting wide-range of NLP tasks from research to industrial applications, including Neural Search, Question Answering, Information Extraction and Sentiment Analysis end-to-end system.",
        long_description=read("README_en.md"),
        long_description_content_type="text/markdown",
        url="https://github.com/PaddlePaddle/PaddleNLP",
        license_files=("LICENSE",),
        packages=setuptools.find_packages(
            where=".",
            exclude=("examples*", "tests*", "applications*", "fast_tokenizer*", "fast_generation*", "model_zoo*"),
        ),
        package_data={
            "paddlenlp.ops": get_package_data_files(
                "paddlenlp.ops", ["CMakeLists.txt", "README.md", "cmake", "fast_transformer", "patches", "optimizer"]
            ),
            "paddlenlp.transformers.layoutxlm": get_package_data_files(
                "paddlenlp.transformers.layoutxlm", ["visual_backbone.yaml"]
            ),
        },
        setup_requires=["cython", "numpy"],
        install_requires=REQUIRED_PACKAGES,
        entry_points={"console_scripts": ["paddlenlp = paddlenlp.cli:main"]},
        extras_require=extras,
        python_requires=">=3.6",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        license="Apache 2.0",
    )
except Exception as e:
    git_checkout(paddlenlp_dir, "paddlenlp/version/__init__.py") if commit != "unknown" else None
    raise e

git_checkout(paddlenlp_dir, "paddlenlp/version/__init__.py") if commit != "unknown" else None
