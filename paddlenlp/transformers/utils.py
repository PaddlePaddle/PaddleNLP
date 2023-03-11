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
import importlib
import inspect
import os
import warnings
from typing import TYPE_CHECKING, Optional, Type

import paddle

if TYPE_CHECKING:
    from paddlenlp.transformers import PretrainedModel

from paddle.nn import Layer

from paddlenlp.utils.env import HF_CACHE_HOME, MODEL_HOME
from paddlenlp.utils.import_utils import import_module
from paddlenlp.utils.log import logger

WEIGHTS_INDEX_NAME = "model_state.pdparams.index.json"
PT_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
WEIGHTS_NAME = "model_state.pdparams"


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


def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    local_files_only=False,
    use_auth_token=None,
    user_agent=None,
    revision=None,
    subfolder="",
    _commit_hash=None,
):
    """
    For a given model:
    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.
    For the description of each arg, see [`PreTrainedModel.from_pretrained`]. `index_filename` is the full path to the
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
        pretrained_model_name_or_path, shard_filenames[-1], cache_dir=cache_dir, revision=_commit_hash
    )
    show_progress_bar = last_shard is None or force_download
    for shard_filename in tqdm(shard_filenames, desc="Downloading shards", disable=not show_progress_bar):
        try:
            # Load from URL
            cached_filename = cached_file(
                pretrained_model_name_or_path,
                shard_filename,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=_commit_hash,
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


def is_safetensors_available():
    return importlib.util.find_spec("safetensors") is not None


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


def paddlenlp_load(path, return_numpy=False):
    if return_numpy:
        with device_guard():
            return paddle.load(path)
    else:
        return paddle.load(path, return_numpy=return_numpy)
