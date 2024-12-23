# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import shutil
import sys
import textwrap
import warnings
from pathlib import Path

from setuptools import find_packages, setup

version_range_max = max(sys.version_info[1], 10) + 1


def read_requirements_file(filepath):
    with open(filepath) as fin:
        requirements = fin.read()
    return requirements


def write_custom_op_api_py(libname, filename):
    libname = str(libname)
    filename = str(filename)
    import paddle

    op_names = paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(libname)
    api_content = [paddle.utils.cpp_extension.extension_utils._custom_api_content(op_name) for op_name in op_names]
    dirname = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    _stub_template = textwrap.dedent(
        """
        # THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY

        {custom_api}

        import os
        import sys
        import types
        import paddle
        import importlib.abc
        import importlib.util

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(cur_dir, "lib/{resource}")

        def __bootstrap__():
            assert os.path.exists(so_path)
            # load custom op shared library with abs path
            custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)

            if os.name == 'nt' or sys.platform.startswith('darwin'):
                # Cpp Extension only support Linux now
                mod = types.ModuleType(__name__)
            else:
                try:
                    spec = importlib.util.spec_from_file_location(__name__, so_path)
                    assert spec is not None
                    mod = importlib.util.module_from_spec(spec)
                    assert isinstance(spec.loader, importlib.abc.Loader)
                    spec.loader.exec_module(mod)
                except ImportError:
                    mod = types.ModuleType(__name__)

            for custom_op in custom_ops:
                setattr(mod, custom_op, eval(custom_op))

        __bootstrap__()

        """
    ).lstrip()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(_stub_template.format(resource=os.path.basename(libname), custom_api="\n\n".join(api_content)))


if len(sys.argv) > 0:
    # generate lib files
    lib_path = Path("src/paddlenlp_kernel/cuda/lib")
    if lib_path.exists():
        shutil.rmtree(lib_path)
    lib_path.mkdir(exist_ok=True)
    (lib_path / "__init__.py").touch(exist_ok=True)
    has_built = False
    for so_file in Path("csrc").glob("**/*.so"):
        so_filename = so_file.name
        # so file
        new_so_filename = so_filename.replace(".so", "_pd.so")
        new_so_file = lib_path / new_so_filename
        # py file
        py_filename = so_filename.replace(".so", ".py")
        new_py_file = lib_path.parent / py_filename
        shutil.copyfile(so_file, new_so_file)
        write_custom_op_api_py(new_so_file, new_py_file)
        has_built = True

    if not has_built:
        warnings.warn("No cuda lib found. Please build cuda lib first. See details in csrc/README.md.")

# NEW ADDED END
setup(
    name="paddlenlp_kernel",
    version="0.1.0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="PaddleNLP GPU OPS cuda & triton.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="paddlenlp kernel contain cuda & triton",
    license="Apache 2.0 License",
    author="PaddlePaddle",
    author_email="paddlenlp@baidu.com",
    url="https://github.com/PaddlePaddle/paddlenlp/ops",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"paddlenlp_kernel.cuda.lib": ["*.so", "*.dll", "*.dylib"]},
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=read_requirements_file("requirements.txt"),
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, version_range_max)],
)
