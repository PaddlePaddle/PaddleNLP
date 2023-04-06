# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""
Import utilities: Utilities related to imports and our lazy inits.
"""
import importlib.util
import operator as op
import os
import sys
from collections import OrderedDict
from typing import Union

from packaging.version import Version, parse

from . import logging

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_PADDLE = os.environ.get("USE_PADDLE", "AUTO").upper()
USE_SAFETENSORS = os.environ.get("USE_SAFETENSORS", "AUTO").upper()

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

_paddle_version = "N/A"
if USE_PADDLE in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _paddle_available = importlib.util.find_spec("paddle") is not None
    _ppxformers_available = False
    if _paddle_available:
        try:
            import paddle

            _paddle_version = paddle.__version__
            logger.info(f"Paddle version {_paddle_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _paddle_available = False

        if _paddle_available:
            try:
                from paddle.incubate.nn.memory_efficient_attention import (
                    memory_efficient_attention,
                )

                memory_efficient_attention
                _ppxformers_available = True
            except ImportError:
                _ppxformers_available = False

else:
    logger.info("Disabling Paddle because USE_PADDLE is set")
    _paddle_available = False
    _ppxformers_available = False

_torch_version = "N/A"
_torch_available = importlib.util.find_spec("torch") is not None
if _torch_available:
    try:
        _torch_version = importlib_metadata.version("torch")
        logger.info(f"PyTorch version {_torch_version} available.")
    except importlib_metadata.PackageNotFoundError:
        _torch_available = False

if USE_SAFETENSORS in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _safetensors_available = importlib.util.find_spec("safetensors") is not None
    if _safetensors_available:
        try:
            _safetensors_version = importlib_metadata.version("safetensors")
            logger.info(f"Safetensors version {_safetensors_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _safetensors_available = False
else:
    logger.info("Disabling Safetensors because USE_TF is set")
    _safetensors_available = False

_transformers_available = importlib.util.find_spec("transformers") is not None
try:
    _transformers_version = importlib_metadata.version("transformers")
    logger.debug(f"Successfully imported transformers version {_transformers_version}")
except importlib_metadata.PackageNotFoundError:
    _transformers_available = False


_inflect_available = importlib.util.find_spec("inflect") is not None
try:
    _inflect_version = importlib_metadata.version("inflect")
    logger.debug(f"Successfully imported inflect version {_inflect_version}")
except importlib_metadata.PackageNotFoundError:
    _inflect_available = False


_unidecode_available = importlib.util.find_spec("unidecode") is not None
try:
    _unidecode_version = importlib_metadata.version("unidecode")
    logger.debug(f"Successfully imported unidecode version {_unidecode_version}")
except importlib_metadata.PackageNotFoundError:
    _unidecode_available = False

_fastdeploy_version = "N/A"
_fastdeploy_available = importlib.util.find_spec("fastdeploy") is not None
if _fastdeploy_available:
    candidates = ("fastdeploy_gpu_python", "fastdeploy_python")
    # For the metadata, we have to look for both fastdeploy_python and fastdeploy_gpu_python
    for pkg in candidates:
        try:
            _fastdeploy_version = importlib_metadata.version(pkg)
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    _fastdeploy_available = _fastdeploy_version != "N/A"
    if _fastdeploy_available:
        logger.debug(f"Successfully imported fastdeploy version {_fastdeploy_version}")

_paddlenlp_available = importlib.util.find_spec("paddlenlp") is not None
try:
    _paddlenlp_version = importlib_metadata.version("paddlenlp")
    logger.debug(f"Successfully imported paddlenlp version {_paddlenlp_version}")
except importlib_metadata.PackageNotFoundError:
    _paddlenlp_available = False

_scipy_available = importlib.util.find_spec("scipy") is not None
try:
    _scipy_version = importlib_metadata.version("scipy")
    logger.debug(f"Successfully imported scipy version {_scipy_version}")
except importlib_metadata.PackageNotFoundError:
    _scipy_available = False

_librosa_available = importlib.util.find_spec("librosa") is not None
try:
    _librosa_version = importlib_metadata.version("librosa")
    logger.debug(f"Successfully imported librosa version {_librosa_version}")
except importlib_metadata.PackageNotFoundError:
    _librosa_available = False

_k_diffusion_available = importlib.util.find_spec("k_diffusion") is not None
try:
    _k_diffusion_version = importlib_metadata.version("k_diffusion")
    logger.debug(f"Successfully imported k-diffusion version {_k_diffusion_version}")
except importlib_metadata.PackageNotFoundError:
    _k_diffusion_available = False

_wandb_available = importlib.util.find_spec("wandb") is not None
try:
    _wandb_version = importlib_metadata.version("wandb")
    logger.debug(f"Successfully imported wandb version {_wandb_version }")
except importlib_metadata.PackageNotFoundError:
    _wandb_available = False

_omegaconf_available = importlib.util.find_spec("omegaconf") is not None
try:
    _omegaconf_version = importlib_metadata.version("omegaconf")
    logger.debug(f"Successfully imported omegaconf version {_omegaconf_version}")
except importlib_metadata.PackageNotFoundError:
    _omegaconf_available = False

_tensorboard_available = importlib.util.find_spec("tensorboard")
try:
    _tensorboard_version = importlib_metadata.version("tensorboard")
    logger.debug(f"Successfully imported tensorboard version {_tensorboard_version}")
except importlib_metadata.PackageNotFoundError:
    _tensorboard_available = False

_visualdl_available = importlib.util.find_spec("visualdl")
try:
    _visualdl_version = importlib_metadata.version("visualdl")
    logger.debug(f"Successfully imported visualdl version {_visualdl_version}")
except importlib_metadata.PackageNotFoundError:
    _visualdl_available = False


def is_paddle_available():
    return _paddle_available


def is_paddlenlp_available():
    return _paddlenlp_available


def is_visualdl_available():
    return _visualdl_available


def is_fastdeploy_available():
    return _fastdeploy_available


def is_ppxformers_available():
    return _ppxformers_available


def is_torch_available():
    return _torch_available


def is_safetensors_available():
    return _safetensors_available


def is_transformers_available():
    return _transformers_available


def is_inflect_available():
    return _inflect_available


def is_unidecode_available():
    return _unidecode_available


def is_scipy_available():
    return _scipy_available


def is_librosa_available():
    return _librosa_available


def is_k_diffusion_available():
    return False  # _k_diffusion_available


def is_wandb_available():
    return _wandb_available


def is_omegaconf_available():
    return _omegaconf_available


def is_tensorboard_available():
    return _tensorboard_available


# docstyle-ignore
FASTDEPLOY_IMPORT_ERROR = """
{0} requires the fastdeploy library but it was not found in your environment. You can install it with pip: `pip install
fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html`
"""

# docstyle-ignore
PADDLE_IMPORT_ERROR = """
{0} requires the Paddle library but it was not found in your environment. Checkout the instructions on the
installation page: https://www.paddlepaddle.org.cn/install/quick and follow the ones that match your environment.
"""

# docstyle-ignore
PPXFORMERS_IMPORT_ERROR = """
{0} requires the scaled_dot_product_attention but your PaddlePaddle donot have this. Checkout the instructions on the
installation page: https://www.paddlepaddle.org.cn/install/quick and follow the ones that match your environment.
"""

# docstyle-ignore
PADDLENLP_IMPORT_ERROR = """
{0} requires the paddlenlp library but it was not found in your environment. You can install it with pip: `pip
install paddlenlp`
"""

# docstyle-ignore
TENSORBOARD_IMPORT_ERROR = """
{0} requires the tensorboard library but it was not found in your environment. You can install it with pip: `pip
install tensorboard`
"""

# docstyle-ignore
VISUALDL_IMPORT_ERROR = """
{0} requires the visualdl library but it was not found in your environment. You can install it with pip: `pip
install visualdl`
"""

# docstyle-ignore
INFLECT_IMPORT_ERROR = """
{0} requires the inflect library but it was not found in your environment. You can install it with pip: `pip install
inflect`
"""

# docstyle-ignore
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""


# docstyle-ignore
SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip: `pip install
scipy`
"""

# docstyle-ignore
LIBROSA_IMPORT_ERROR = """
{0} requires the librosa library but it was not found in your environment.  Checkout the instructions on the
installation page: https://librosa.org/doc/latest/install.html and follow the ones that match your environment.
"""


# docstyle-ignore
UNIDECODE_IMPORT_ERROR = """
{0} requires the unidecode library but it was not found in your environment. You can install it with pip: `pip install
Unidecode`
"""

# docstyle-ignore
K_DIFFUSION_IMPORT_ERROR = """
{0} requires the k-diffusion library but it was not found in your environment. You can install it with pip: `pip
install k-diffusion`
"""

# docstyle-ignore
WANDB_IMPORT_ERROR = """
{0} requires the wandb library but it was not found in your environment. You can install it with pip: `pip
install wandb`
"""

# docstyle-ignore
OMEGACONF_IMPORT_ERROR = """
{0} requires the omegaconf library but it was not found in your environment. You can install it with pip: `pip
install omegaconf`
"""

# docstyle-ignore
TENSORBOARD_IMPORT_ERROR = """
{0} requires the tensorboard library but it was not found in your environment. You can install it with pip: `pip
install tensorboard`
"""

BACKENDS_MAPPING = OrderedDict(
    [
        ("fastdeploy", (is_fastdeploy_available, FASTDEPLOY_IMPORT_ERROR)),
        ("paddle", (is_paddle_available, PADDLE_IMPORT_ERROR)),
        ("paddlenlp", (is_paddlenlp_available, PADDLENLP_IMPORT_ERROR)),
        ("visualdl", (is_visualdl_available, VISUALDL_IMPORT_ERROR)),
        ("inflect", (is_inflect_available, INFLECT_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("unidecode", (is_unidecode_available, UNIDECODE_IMPORT_ERROR)),
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        ("k_diffusion", (is_k_diffusion_available, K_DIFFUSION_IMPORT_ERROR)),
        ("wandb", (is_wandb_available, WANDB_IMPORT_ERROR)),
        ("omegaconf", (is_omegaconf_available, OMEGACONF_IMPORT_ERROR)),
        ("tensorboard", (_tensorboard_available, TENSORBOARD_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))

    if name in [
        "VersatileDiffusionTextToImagePipeline",
        "VersatileDiffusionPipeline",
        "VersatileDiffusionDualGuidedPipeline",
        "StableDiffusionImageVariationPipeline",
        "UnCLIPPipeline",
    ] and is_paddlenlp_version("<", "2.5.0"):
        raise ImportError(
            f"You need to install `paddlenlp>=2.5.0` in order to use {name}: \n```\n pip install"
            " --upgrade paddlenlp \n```"
        )

    if name in ["StableDiffusionDepth2ImgPipeline", "StableDiffusionPix2PixZeroPipeline"] and is_paddlenlp_version(
        "<", "2.5.1"  # TODO version
    ):
        raise ImportError(
            f"You need to install `paddlenlp>=2.5.1` in order to use {name}: \n```\n pip install"
            " --upgrade paddlenlp \n```"
        )


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        if key.startswith("_"):
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    Args:
    Compares a library version to some requirement using a given operation.
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L338
def is_torch_version(operation: str, version: str):
    """
    Args:
    Compares the current PyTorch version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    return compare_versions(parse(_torch_version), operation, version)


def is_paddle_version(operation: str, version: str):
    """
    Args:
    Compares the current Paddle version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of Paddle
    """
    return compare_versions(parse(_paddle_version), operation, version)


def is_paddlenlp_version(operation: str, version: str):
    """
    Args:
    Compares the current paddlenlp version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _paddlenlp_available:
        return False
    return compare_versions(parse(_paddlenlp_version), operation, version)


def is_k_diffusion_version(operation: str, version: str):
    """
    Args:
    Compares the current k-diffusion version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _k_diffusion_available:
        return False
    return compare_versions(parse(_k_diffusion_version), operation, version)


class OptionalDependencyNotAvailable(BaseException):
    """An error indicating that an optional dependency of Diffusers was not found in the environment."""
