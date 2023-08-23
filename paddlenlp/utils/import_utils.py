# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from __future__ import annotations

import importlib.util
import os
import shutil
import site
import sys
from typing import Optional, Type

import pip

from paddlenlp.utils.log import logger


def is_datasets_available():
    import importlib

    return importlib.util.find_spec("datasets") is not None


def is_paddle_cuda_available() -> bool:
    if is_paddle_available():
        import paddle

        return paddle.device.cuda.device_count() > 0
    else:
        return False


def is_paddle_available() -> bool:
    """check if `torch` package is installed
    Returns:
        bool: if `torch` is available
    """
    return is_package_available("paddle")


def is_psutil_available():
    return importlib.util.find_spec("psutil") is not None


def is_tiktoken_available():
    return importlib.util.find_spec("tiktoken") is not None


def is_torch_available() -> bool:
    """check if `torch` package is installed
    Returns:
        bool: if `torch` is available
    """
    return is_package_available("torch")


def is_package_available(package_name: str) -> bool:
    """check if the package is avaliable
    Args:
        package_name (str): the installed package name
    Returns:
        bool: the existence of installed package
    """
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None and package_spec.has_location


def is_fast_tokenizer_available() -> bool:
    """check if `fast_tokenizer` ia avaliable
    Returns:
        bool: if `fast_tokenizer` is avaliable
    """
    return is_package_available("fast_tokenizer")


def is_paddlenlp_ops_available() -> bool:
    """check if `paddlenlp_ops` ia avaliable
    Returns:
        bool: if `fast_tokenizer` is avaliable
    """
    return is_package_available("paddlenlp_ops")


def is_transformers_available() -> bool:
    """check if `transformers` package is installed
    Returns:
        bool: if `transformers` is available
    """
    return is_package_available("transformers")


def install_package(
    package_name: str,
    version: Optional[str] = None,
    module_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """install the specific version of package

    Args:
        package_name (str): the name of package
        version (str): the version of package
        module_name (str): the imported name of package
        cache_dir (str): cache dir
    """
    module_name = module_name or package_name

    # 1. remove the existing version of package
    uninstall_package(package_name, module_name)

    # 2. install the package
    if version:
        package_name += f"=={version}"

    arguments = ["install"]
    if cache_dir:
        arguments += ["-t", cache_dir]
        sys.path.insert(0, cache_dir)

    # 3. load the pypi mirror to speedup of installing packages
    mirror_key = "PYPI_MIRROR"
    mirror_source = os.environ.get(mirror_key, None)
    if mirror_source is not None:
        logger.info(f"loading <{mirror_source}> from as the final mirror source to install package.")
        arguments += ["-i", mirror_source]

    arguments += [package_name]
    pip.main(arguments)

    # 4. add site-package to the top of package
    for site_package_dir in site.getsitepackages():
        sys.path.insert(0, site_package_dir)


def uninstall_package(package_name: str, module_name: Optional[str] = None):
    """uninstall the pacakge from site-packages.

    To remove the cache of source package module & class & method, it should:
        1. remove the source files of packages under the `site-packages` dir.
        2. remove the cache under the `locals()`
        3. remove the cache under the `sys.modules`

    Args:
        package_name (str): the name of package
    """
    module_name = module_name or package_name
    for site_package_dir in site.getsitepackages():
        for file in os.listdir(site_package_dir):
            package_dir = os.path.join(site_package_dir, file)
            if file.startswith(package_name) and os.path.isdir(package_dir):
                shutil.rmtree(package_dir)

    for site_package_dir in site.getsitepackages():
        while sys.path[0] == site_package_dir:
            sys.path.pop(0)

    for key in list(locals().keys()):
        if module_name in key:
            del locals()[key]

    for key in list(sys.modules.keys()):
        if module_name in key:
            del sys.modules[key]


def import_module(module_name: str) -> Optional[Type]:
    """import moudle base on the model
    Args:
        module_name (str): the name of target module
    """
    # 1. prepare the name
    assert "." in module_name, "`.` must be in the module_name"
    index = module_name.rindex(".")
    module = module_name[:index]
    target_module_name = module_name[index + 1 :]

    # 2. get the target module name
    try:
        module = importlib.import_module(module)
        target_module = getattr(module, target_module_name, None)
        return target_module
    except ModuleNotFoundError:
        return None
