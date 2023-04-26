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
from __future__ import annotations

import contextlib
import functools
import inspect
import os
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, ContextManager, List, Optional, Type, Union

import numpy as np

from paddlenlp.utils.downloader import (
    COMMUNITY_MODEL_PREFIX,
    download_check,
    get_path_from_url_with_filelock,
    url_file_exists,
)

if TYPE_CHECKING:
    from paddlenlp.transformers import PretrainedModel

import paddle
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.utils import EntryNotFoundError
from paddle.nn import Layer
from requests.exceptions import HTTPError
from tqdm import tqdm

from paddlenlp.utils.env import HF_CACHE_HOME, MODEL_HOME
from paddlenlp.utils.import_utils import import_module, is_safetensors_available
from paddlenlp.utils.log import logger
from paddlenlp.utils.serialization import load_torch

if is_safetensors_available():

    from safetensors import safe_open
    from safetensors.paddle import load_file as safe_load_file


def map_numpy_dtype(dtype: np.dtype) -> paddle.dtype:
    """map the numpy dtype to paddle dtype, eg: np.float32 -> paddle.float32

    Args:
        dtype (np.dtype): the dtype of numpy

    Returns:
        paddle.dtype: the dtype of paddle tensor
    """
    if not isinstance(dtype, np.dtype):
        raise ValueError(f"the dtype should be numpy.dtype, but get {type(dtype)}")

    mappings = {
        # floating point
        "float64": paddle.float64,
        "float32": paddle.float32,
        "float16": paddle.float16,
        "uint16": paddle.bfloat16,
        # int point
        "int64": paddle.int64,
        "int32": paddle.int32,
        "int16": paddle.int16,
        "bool": paddle.bool,
    }

    if dtype.name not in mappings:
        raise ValueError(f"dtype must be in list of dict types<{','.join(list(mappings.keys()))}>, but got {dtype}")

    return mappings[dtype.name]


def get_tensor_dtype(tensor: Union[paddle.Tensor, np.ndarray], return_string: bool = True) -> Union[str, paddle.dtype]:
    """get the dtype of

    Args:
        tensor (dict[str, Union[paddle.Tensor, np.ndarray]]): get the dtype
        return_string (bool, optional): _description_. Defaults to True.

    Returns:
        Union[str, paddle.dtype]: the target dtpe object
    """
    if paddle.is_tensor(tensor):
        dtype = tensor.dtype
    else:
        assert isinstance(tensor, np.ndarray), f"tensor must be paddle tensor or numpy ndarray, got {type(tensor)}!"
        dtype = map_numpy_dtype(tensor.dtype)
    return dtype


HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"


def fn_args_to_dict(func, *args, **kwargs):
    """
    Inspect function `func` and its arguments for running, and extract a
    dict mapping between argument names and keys.
    """
    if hasattr(inspect, "getfullargspec"):
        (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _, _) = inspect.getfullargspec(func)
    else:
        (spec_args, spec_varargs, spec_varkw, spec_defaults) = inspect.getargspec(func)
    # add positional argument values
    init_dict = dict(zip(spec_args, args))
    # add default argument values
    kwargs_dict = dict(zip(spec_args[-len(spec_defaults) :], spec_defaults)) if spec_defaults else {}
    for k in list(kwargs_dict.keys()):
        if k in init_dict:
            kwargs_dict.pop(k)
    kwargs_dict.update(kwargs)
    init_dict.update(kwargs_dict)
    return init_dict


def adapt_stale_fwd_patch(self, name, value):
    """
    Since there are some monkey patches for forward of PretrainedModel, such as
    model compression, we make these patches compatible with the latest forward
    method.
    """
    if name == "forward":
        # NOTE(guosheng): In dygraph to static, `layer.forward` would be patched
        # by an instance of `StaticFunction`. And use string compare to avoid to
        # import fluid.
        if type(value).__name__.endswith("StaticFunction"):
            return value
        if hasattr(inspect, "getfullargspec"):
            (
                patch_spec_args,
                patch_spec_varargs,
                patch_spec_varkw,
                patch_spec_defaults,
                _,
                _,
                _,
            ) = inspect.getfullargspec(value)
            (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _, _) = inspect.getfullargspec(self.forward)
        else:
            (patch_spec_args, patch_spec_varargs, patch_spec_varkw, patch_spec_defaults) = inspect.getargspec(value)
            (spec_args, spec_varargs, spec_varkw, spec_defaults) = inspect.getargspec(self.forward)
        new_args = [
            arg
            for arg in ("output_hidden_states", "output_attentions", "return_dict")
            if arg not in patch_spec_args and arg in spec_args
        ]

        if new_args:
            if self.__module__.startswith("paddlenlp"):
                warnings.warn(
                    f"The `forward` method of {self.__class__ if isinstance(self, Layer) else self} is patched and the patch "
                    "might be based on an old oversion which missing some "
                    f"arguments compared with the latest, such as {new_args}. "
                    "We automatically add compatibility on the patch for "
                    "these arguemnts, and maybe the patch should be updated."
                )
            else:
                warnings.warn(
                    f"The `forward` method of {self.__class__ if isinstance(self, Layer) else self} "
                    "is patched and the patch might be conflict with patches made "
                    f"by paddlenlp which seems have more arguments such as {new_args}. "
                    "We automatically add compatibility on the patch for "
                    "these arguemnts, and maybe the patch should be updated."
                )
            if isinstance(self, Layer) and inspect.isfunction(value):

                @functools.wraps(value)
                def wrap_fwd(*args, **kwargs):
                    for arg in new_args:
                        kwargs.pop(arg, None)
                    return value(self, *args, **kwargs)

            else:

                @functools.wraps(value)
                def wrap_fwd(*args, **kwargs):
                    for arg in new_args:
                        kwargs.pop(arg, None)
                    return value(*args, **kwargs)

            return wrap_fwd
    return value


class InitTrackerMeta(type(Layer)):
    """
    This metaclass wraps the `__init__` method of a class to add `init_config`
    attribute for instances of that class, and `init_config` use a dict to track
    the initial configuration. If the class has `_pre_init` or `_post_init`
    method, it would be hooked before or after `__init__` and called as
    `_pre_init(self, init_fn, init_args)` or `_post_init(self, init_fn, init_args)`.
    Since InitTrackerMeta would be used as metaclass for pretrained model classes,
    which always are Layer and `type(Layer)` is not `type`, thus use `type(Layer)`
    rather than `type` as base class for it to avoid inheritance metaclass
    conflicts.
    """

    def __init__(cls, name, bases, attrs):
        init_func = cls.__init__
        # If attrs has `__init__`, wrap it using accessable `_pre_init, _post_init`.
        # Otherwise, no need to wrap again since the super cls has been wraped.
        # TODO: remove reduplicated tracker if using super cls `__init__`
        pre_init_func = getattr(cls, "_pre_init", None) if "__init__" in attrs else None
        post_init_func = getattr(cls, "_post_init", None) if "__init__" in attrs else None
        cls.__init__ = InitTrackerMeta.init_and_track_conf(init_func, pre_init_func, post_init_func)
        super(InitTrackerMeta, cls).__init__(name, bases, attrs)

    @staticmethod
    def init_and_track_conf(init_func, pre_init_func=None, post_init_func=None):
        """
        wraps `init_func` which is `__init__` method of a class to add `init_config`
        attribute for instances of that class.
        Args:
            init_func (callable): It should be the `__init__` method of a class.
                warning: `self` always is the class type of down-stream model, eg: BertForTokenClassification
            pre_init_func (callable, optional): If provided, it would be hooked after
                `init_func` and called as `pre_init_func(self, init_func, *init_args, **init_args)`.
                Default None.
            post_init_func (callable, optional): If provided, it would be hooked after
                `init_func` and called as `post_init_func(self, init_func, *init_args, **init_args)`.
                Default None.

        Returns:
            function: the wrapped function
        """

        @functools.wraps(init_func)
        def __impl__(self, *args, **kwargs):
            # registed helper by `pre_init_func`
            if pre_init_func:
                pre_init_func(self, init_func, *args, **kwargs)
            # keep full configuration
            init_func(self, *args, **kwargs)
            # registed helper by `post_init_func`
            if post_init_func:
                post_init_func(self, init_func, *args, **kwargs)
            self.init_config = kwargs
            if args:
                kwargs["init_args"] = args
            kwargs["init_class"] = self.__class__.__name__

        return __impl__

    def __setattr__(self, name, value):
        value = adapt_stale_fwd_patch(self, name, value)
        return super(InitTrackerMeta, self).__setattr__(name, value)


def param_in_func(func, param_field: str) -> bool:
    """check if the param_field is in `func` method, eg: if the `bert` param is in `__init__` method

    Args:
        cls (type): the class of PretrainedModel
        param_field (str): the name of field

    Returns:
        bool: the result of existence
    """

    if hasattr(inspect, "getfullargspec"):
        result = inspect.getfullargspec(func)
    else:
        result = inspect.getargspec(func)

    return param_field in result[0]


def resolve_cache_dir(pretrained_model_name_or_path: str, from_hf_hub: bool, cache_dir: Optional[str] = None) -> str:
    """resolve cache dir for PretrainedModel and PretrainedConfig

    Args:
        pretrained_model_name_or_path (str): the name or path of pretrained model
        from_hf_hub (bool): if load from huggingface hub
        cache_dir (str): cache_dir for models
    """
    if os.path.isdir(pretrained_model_name_or_path):
        return pretrained_model_name_or_path

    # hf hub library takes care of appending the model name so we don't append the model name
    if from_hf_hub:
        if cache_dir is not None:
            return cache_dir
        else:
            return HF_CACHE_HOME
    else:
        if cache_dir is not None:
            # since model_clas.from_pretrained calls config_clas.from_pretrained, the model_name may get appended twice
            if cache_dir.endswith(pretrained_model_name_or_path):
                return cache_dir
            else:
                return os.path.join(cache_dir, pretrained_model_name_or_path)
        return os.path.join(MODEL_HOME, pretrained_model_name_or_path)


def find_transformer_model_type(model_class: Type) -> str:
    """get the model type from module name,
        eg:
            BertModel -> bert,
            RobertaForTokenClassification -> roberta

    Args:
        model_class (Type): the class of model

    Returns:
        str: the type string
    """
    from paddlenlp.transformers import PretrainedModel

    default_model_type = ""

    if not issubclass(model_class, PretrainedModel):
        return default_model_type

    module_name: str = model_class.__module__
    if not module_name.startswith("paddlenlp.transformers."):
        return default_model_type

    tokens = module_name.split(".")
    if len(tokens) < 3:
        return default_model_type

    return tokens[2]


def find_transformer_model_class_by_name(model_name: str) -> Optional[Type[PretrainedModel]]:
    """find transformer model_class by name

    Args:
        model_name (str): the string of class name

    Returns:
        Optional[Type[PretrainedModel]]: optional pretrained-model class
    """
    transformer_module = import_module("paddlenlp.transformers")

    for obj_name in dir(transformer_module):
        if obj_name.startswith("_"):
            continue
        obj = getattr(transformer_module, obj_name, None)
        if obj is None:
            continue

        name = getattr(obj, "__name__", None)
        if name is None:
            continue

        if name == model_name:
            return obj
    logger.debug(f"can not find model_class<{model_name}>")
    return None


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).
    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.
    Example:
    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError("`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


def paddlenlp_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
) -> str:

    # check in cache_dir
    weight_file_path = os.path.join(cache_dir, filename)
    if os.path.exists(weight_file_path):
        logger.info(f"Already cached {weight_file_path}")
        return weight_file_path

    # find in community repo
    community_model_file_path = os.path.join(COMMUNITY_MODEL_PREFIX, repo_id, subfolder, filename)

    # check wether the target file exist in the comunity bos server
    if url_file_exists(community_model_file_path):
        weight_file_path = get_path_from_url_with_filelock(community_model_file_path, cache_dir)
        download_check(community_model_file_path, "paddlenlp_hub_download")
        return weight_file_path

    return None


# Return value when trying to load a file from cache but the file does not exist in the distant repo.
_CACHED_NO_EXIST = object()


def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    subfolder="",
):
    """
    For a given model:
    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.
    For the description of each arg, see [`PretrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """

    import json

    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub
    cached_filenames = []
    # Check if the model is already cached or not. We only try the last checkpoint, this should cover most cases of
    # downloaded (if interrupted).
    last_shard = try_to_load_from_cache(
        pretrained_model_name_or_path,
        shard_filenames[-1],
        cache_dir=cache_dir,
    )

    show_progress_bar = last_shard is None
    for shard_filename in tqdm(shard_filenames, desc="Downloading shards", disable=not show_progress_bar):
        try:
            cached_filename = paddlenlp_hub_download(
                pretrained_model_name_or_path,
                shard_filename,
                subfolder=subfolder,
                cache_dir=cache_dir,
            )
        # We have already dealt with RepositoryNotFoundError and RevisionNotFoundError when getting the index, so
        # we don't have to catch them here.
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {shard_filename} which is "
                "required according to the checkpoint index."
            )
        except HTTPError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load {shard_filename}. You should try"
                " again after checking your internet connection."
            )

        cached_filenames.append(cached_filename)

    return cached_filenames, sharded_metadata


@contextlib.contextmanager
def device_guard(device="cpu", dev_id=0):
    origin_device = paddle.device.get_device()
    if device == "cpu":
        paddle.set_device(device)
    elif device in ["gpu", "xpu", "npu"]:
        paddle.set_device("{}:{}".format(device, dev_id))
    try:
        yield
    finally:
        paddle.set_device(origin_device)


def check_map_location(map_location: str) -> tuple[str, int]:
    """check whether the map_location is valid

    Args:
        location (str): the location of device

    Returns:
        tuple[str, int]: [device name, device_id]
    """
    dev_id = 0
    if map_location.startswith("gpu"):
        if map_location[3:]:
            dev_id = int(map_location[3:])
            map_location = "gpu"

    assert map_location in [
        "cpu",
        "gpu",
        "gpu:{id}",
        "xpu",
        "npu",
        "numpy",
    ], "the value of map_location should be one of [`cpu`, `gpu`, `gpu:id`, `xpu`, `npu`, `numpy`]"
    return map_location, dev_id


def paddlenlp_load(path: str, map_location: str = "cpu") -> dict[str, paddle.Tensor]:
    """load weight as state dict as tensor to the target device

    Args:
        path (str): the path of weight file
        map_location(str): the location of target weight file

    Returns:
        Dict[str, paddle.Tensor]: the state dict from weight file
    """
    map_location, dev_id = check_map_location(map_location)
    with device_guard(map_location, dev_id=dev_id):
        state_dict = paddle.load(path)
    return state_dict


def is_paddle_support_lazy_init():
    return hasattr(paddle, "LazyGuard")


def get_state_dict_dtype(state_dict: dict[str, Union[paddle.Tensor, np.ndarray]]):
    """
    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
    """
    first_key = next(iter(state_dict))
    dtype = state_dict[first_key].dtype
    if isinstance(dtype, paddle.dtype):
        return str(dtype)[7:]
    name = dtype.name
    name = "bfloat16" if name == "uint16" else name
    return name


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike], map_location: str = "cpu"
) -> dict[str, Union[np.ndarray, paddle.Tensor]]:
    """load paddle/pytorch/safetensor weight file as numpy/cpu/gpu or other device

        when map_location is 'numpy', the loaded state dict value is np.ndarray
        when map_location is 'cpu', the loaded state dict value is np.ndarray

    Args:
        checkpoint_file (Union[str, os.PathLike]): the path of weight file
        map_location (str, optional): the target device location of state dict. Defaults to "gpu".

    Returns:
        dict[str, Union[np.ndarray, paddle.Tensor]]: the state dict of weight file
    """
    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        # Check format of the archive
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()

        # safetensor support load pytorch/paddle/numpy state_dict
        if metadata.get("format", None) not in ["pt", "pd", "np"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )

        # TODO(wj-Mcat): current it only support load paddle safetensor
        elif metadata["format"] != "pd":
            raise NotImplementedError(
                f"Conversion from a {metadata['format']} safetensors archive to PaddlePaddle is not implemented yet."
            )
        return safe_load_file(checkpoint_file)

    # if the checkpoint file is pytorch file, then
    if checkpoint_file.endswith(".bin"):
        state_dict = load_torch(checkpoint_file)
        if map_location != "numpy":
            device, device_id = check_map_location(map_location)

            with device_guard(device, device_id):
                for key in state_dict.keys():
                    state_dict[key] = paddle.to_tensor(state_dict[key], get_tensor_dtype(state_dict[key]))

        return state_dict

    elif checkpoint_file.endswith(".pdparams"):
        return paddlenlp_load(checkpoint_file, map_location)

    raise EnvironmentError(
        f"unspported checkpoint file<{checkpoint_file}>, the type of it should be one of [`safetensor`, `pytroch`, `paddle`]"
    )


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)
