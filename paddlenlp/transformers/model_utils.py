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

import copy
import gc
import inspect
import json
import os
import re
import tempfile
import warnings
from contextlib import contextmanager
from functools import partial

# from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import paddle
import paddle.nn as nn
import six
from huggingface_hub import (
    create_repo,
    get_hf_file_metadata,
    hf_hub_url,
    repo_type_and_id_from_hf_id,
    upload_folder,
)
from huggingface_hub.utils import EntryNotFoundError
from paddle import Tensor
from paddle.nn import Embedding, Layer

# TODO(fangzeyang) Temporary fix and replace by paddle framework downloader later
from paddle.utils.download import is_url as is_remote_url
from tqdm.auto import tqdm

from paddlenlp.utils.downloader import get_path_from_url_with_filelock, hf_file_exists
from paddlenlp.utils.env import (
    CONFIG_NAME,
    LEGACY_CONFIG_NAME,
    PADDLE_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    PYTORCH_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)
from paddlenlp.utils.log import logger

from ..utils import device_guard
from .configuration_utils import PretrainedConfig
from .conversion_utils import ConversionMixin
from .generation_utils import GenerationMixin
from .utils import (  # convert_ndarray_dtype,
    ContextManagers,
    InitTrackerMeta,
    adapt_stale_fwd_patch,
    cached_file,
    cached_file_for_hf_hub,
    convert_file_size_to_int,
    dtype_byte_size,
    fn_args_to_dict,
    get_checkpoint_shard_files,
    is_paddle_support_lazy_init,
    is_safetensors_available,
    paddlenlp_load,
    resolve_cache_dir,
    weight_name_suffix,
)

__all__ = [
    "PretrainedModel",
    "register_base_model",
]


def unwrap_optimizer(optimizer, optimizer_instances=()):
    if optimizer is None:
        return None
    while hasattr(optimizer, "_inner_opt") and not isinstance(optimizer, optimizer_instances):
        optimizer = optimizer._inner_opt
    if isinstance(optimizer, optimizer_instances):
        return optimizer
    return None


if is_safetensors_available():

    from safetensors import safe_open
    from safetensors.numpy import load_file as safe_load_file
    from safetensors.numpy import save_file as safe_save_file


def prune_linear_layer(layer: nn.Linear, index: paddle.Tensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.
    Used to remove heads.
    Args:
        layer (`paddle.nn.Linear`): The layer to prune.
        index (`paddle.Tensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.
    Returns:
        `paddle.nn.Linear`: The pruned layer as a new layer with `stop_gradient=False`.
    """
    index = index.to(layer.weight)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.shape)
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias_attr=layer.bias is not None)
    new_layer.weight.stop_gradient = True
    new_layer.weight.copy_(W)
    new_layer.weight.stop_gradient = False
    if layer.bias is not None:
        new_layer.bias.stop_gradient = True
        new_layer.bias.copy_(b)
        new_layer.bias.stop_gradient = False
    return new_layer


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], paddle.Tensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.
    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.
    Returns:
        `Tuple[Set[int], paddle.Tensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = paddle.ones([n_heads, head_size])
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.reshape([-1]).eq(1)
    index: paddle.Tensor = paddle.arange(len(mask))[mask].cast("int64")
    return heads, index


def apply_chunking_to_forward(
    forward_fn: Callable[..., paddle.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> paddle.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.
    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.
    Args:
        forward_fn (`Callable[..., paddle.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[paddle.Tensor]`):
            The input tensors of `forward_fn` which will be chunked
    Returns:
        `paddle.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.
    Examples:
    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, axis=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return paddle.concat(output_chunks, axis=chunk_dim)

    return forward_fn(*input_tensors)


def unwrap_model(model, *args, **kwargs):
    raw_model = model
    while hasattr(raw_model, "_layers") or hasattr(raw_model, "_layer"):
        if hasattr(raw_model, "_layers"):
            # Caused by issue https://github.com/PaddlePaddle/PaddleNLP/issues/5295
            # TODO: remove this after we fix the issue
            if raw_model._layers is None:
                break
            raw_model = raw_model._layers
        else:
            if raw_model._layer is None:
                break
            raw_model = raw_model._layer

    return raw_model


def _add_variant(weights_name: str, variant=None) -> str:
    if variant is not None and len(variant) > 0:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


@contextmanager
def dtype_guard(dtype="float32"):
    origin_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype(dtype)
    try:
        yield
    finally:
        paddle.set_default_dtype(origin_dtype)


_init_weights = True


@contextmanager
def no_init_weights(_enable=True):
    """
    Context manager to globally disable weight initialization to speed up loading large models.

    TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
    """
    global _init_weights
    old_init_weights = _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = old_init_weights


def get_parameter_dtype(parameter: nn.Layer) -> paddle.dtype:
    """get dtype of parameter which should be sub-class of nn.Layer

    Args:
        parameter (nn.Layer): the instance of layer

    Returns:
        paddle.dtype: the dtype of tensor
    """

    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype

    # TODO(wj-Mcat): get dtype of model when it's in DataParallel Mode.
    return last_dtype


def load_state_dict(checkpoint_file: Union[str, os.PathLike], tensor_parallel_split_mapping=None):
    """
    Reads a PaddlePaddle checkpoint file, returning properly formatted errors if they arise.
    """
    if tensor_parallel_split_mapping is None:
        tensor_parallel_split_mapping = {}

    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        # Check format of the archive
        with safe_open(checkpoint_file, framework="np") as f:
            metadata = f.metadata()
        if metadata.get("format") not in ["pd", "np"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        if metadata["format"] == "pd":
            raise ValueError("Currently unsupport paddle weights file, use numpy instead.")
            return safe_load_file(checkpoint_file)
        if metadata["format"] == "np":
            logger.warning("loading safe.")
            state_dict = {}
            with safe_open(checkpoint_file, framework="np") as f:
                for key in f.keys():
                    py_safe_slice_ = f.get_slice(key)
                    if key in tensor_parallel_split_mapping:
                        weight = tensor_parallel_split_mapping[key](py_safe_slice_)
                    else:
                        weight = py_safe_slice_[:]
                    state_dict[key] = weight

            logger.warning("loading done.")
            for k in list(state_dict.keys()):
                with device_guard():
                    state_dict[k] = paddle.Tensor(state_dict.pop(k), zero_copy=True)

            logger.warning("copy paddle tensor done.")
            return state_dict

    state_dict = paddlenlp_load(checkpoint_file, map_location="cpu")
    return state_dict


def resolve_weight_file_from_hf_hub(repo_id: str, cache_dir: str, support_conversion: bool, subfolder=None):
    """find the suitable weight file name

    Args:
        repo_id (str): repo name of huggingface hub
        cache_dir (str): cache dir for hf
        support_conversion (bool): whether support converting pytorch weight file to paddle weight file
        subfolder (str, optional) An optional value corresponding to a folder inside the repo.
    """
    is_local = os.path.isdir(repo_id)
    if not is_local:
        if hf_file_exists(repo_id, PADDLE_WEIGHTS_NAME, subfolder=subfolder):
            file_name = PADDLE_WEIGHTS_NAME
            assert (
                support_conversion is False
            ), "Please call set convert_from_torch for paddle weights on huggingface hub, eg. Model.from_pretrained(model_name, from_hf_hub=True, convert_from_torch=False)"
        elif hf_file_exists(repo_id, PYTORCH_WEIGHTS_NAME, subfolder=subfolder):
            if not support_conversion:
                raise EntryNotFoundError(
                    f"can not download `{PADDLE_WEIGHTS_NAME} from https://huggingface.co/{repo_id}` "
                    "and current model doesn't support conversion from pytorch weight file to paddle weight file"
                )
            file_name = PYTORCH_WEIGHTS_NAME
        else:
            raise EntryNotFoundError(
                message=f"can not find the paddle/pytorch weight file from: https://huggingface.co/{repo_id}",
                response=None,
            )
    else:
        # for local file, we use support_conversion to select paddle or torch weight.
        file_name = PYTORCH_WEIGHTS_NAME if support_conversion else PADDLE_WEIGHTS_NAME

    return cached_file_for_hf_hub(repo_id, file_name, cache_dir, subfolder)


def register_base_model(cls):
    """
    A decorator for `PretrainedModel` class. It first retrieves the parent class
    of the class being decorated, then sets the `base_model_class` attribute
    of that parent class to be the class being decorated. In summary, the decorator registers
    the decorated class as the base model class in all derived classes under the same architecture.

    Args:
        cls (PretrainedModel): The class (inherited from PretrainedModel) to be decorated .

    Returns:
        PretrainedModel: The input class `cls` after decorating.

    Example:
        .. code-block::

            from paddlenlp.transformers import BertModel, register_base_model

            BertModel = register_base_model(BertModel)
            assert BertModel.base_model_class == BertModel
    """
    base_cls = cls.__bases__[0]
    assert issubclass(
        base_cls, PretrainedModel
    ), "`register_base_model` should be used on subclasses of PretrainedModel."
    base_cls.base_model_class = cls
    return cls


class BackboneMixin:
    def forward_with_filtered_kwargs(self, *args, **kwargs):
        signature = dict(inspect.signature(self.forward).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature}

        return self(*args, **filtered_kwargs)


_re_layer_prefix = re.compile(r"\.(\d+)\.")


def _partion_for_pipeline_mode(keys):
    # the keys should be sort in networks order
    # TODO maybe handle tie_weight ?
    def layer_prefix(key):
        ret = _re_layer_prefix.search(key)
        if ret is not None:
            return key[0 : ret.end()]
        return ""

    keys = list(keys)
    start_idx = -1
    prefix_str = None
    parttion_map = {}
    for k in keys:
        prefix = layer_prefix(k)
        if prefix != prefix_str:
            prefix_str = prefix
            start_idx += 1
        parttion_map[k] = start_idx

    # if only one parttion, we don't parttion it
    if start_idx < 1:
        return {keys[i]: i for i in range(len(keys))}

    return parttion_map


def shard_checkpoint(
    state_dict: Dict[str, paddle.Tensor],
    max_shard_size: Union[int, str] = "10GB",
    weights_name: str = PADDLE_WEIGHTS_NAME,
    shard_format="naive",
):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.

    The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no
    optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the
    limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],
    [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger that `max_sahrd_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        state_dict (`Dict[str, paddle.Tensor]`): The state dictionary of a model to save.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
        weights_name (`str`, *optional*, defaults to `"model_state.pdparams"`):
            The name of the model save file.
        shard_format (`str`, *optional*, defaults to `"naive"`):
            support naive or pipeline.
    """
    assert shard_format in [
        "naive",
        "pipeline",
    ], f"Invalid shard_format: {shard_format}, it show be `naive` or `pipeline`."

    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0

    if shard_format == "naive":
        for key, weight in state_dict.items():
            weight_size = weight.numel().item() * dtype_byte_size(weight.dtype)
            # If this weight is going to tip up over the maximal size, we split.
            if current_block_size + weight_size > max_shard_size:
                # fix if the first param is large than max_shard_size
                if len(current_block) > 0:
                    sharded_state_dicts.append(current_block)
                current_block = {}
                current_block_size = 0

            current_block[key] = weight
            current_block_size += weight_size
            total_size += weight_size

        # Add the last block
        sharded_state_dicts.append(current_block)

    if shard_format == "pipeline":
        parttion_map = _partion_for_pipeline_mode(state_dict.keys())
        partition_num = max(parttion_map.values())

        for index in range(partition_num + 1):
            weight_names = [k for k, v in parttion_map.items() if v == index]
            weight_size = sum(
                state_dict[key].numel().item() * dtype_byte_size(state_dict[key].dtype) for key in weight_names
            )

            # try to add new block
            if current_block_size + weight_size > max_shard_size:
                # fix if the first param is large than max_shard_size
                if len(current_block) > 0:
                    sharded_state_dicts.append(current_block)
                current_block = {}
                current_block_size = 0
            for key in weight_names:
                current_block[key] = state_dict[key]
            current_block_size += weight_size
            total_size += weight_size

        # Add the last block
        sharded_state_dicts.append(current_block)
        logger.info(f"The average size of partition is around: {total_size//partition_num}")

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(".pdparams", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.pdparams")
        shard_file = shard_file.replace(
            ".safetensors", f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors"
        )
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index


def load_sharded_checkpoint(model, folder, variant=None, strict=True, prefer_safe=False):
    """
    This is the same as [`paddle.nn.Layer.set_state_dict`]
    but for a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`paddle.nn.Module`): The model in which to load the checkpoint.
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        variant (`str`): The model variant.
        strict (`bool`, *optional`, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
        prefer_safe (`bool`, *optional*, defaults to `False`):
            If both safetensors and Paddle save files are present in checkpoint and `prefer_safe` is True, the safetensors
            files will be loaded. Otherwise, Paddle files are always loaded when possible.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """
    # Load the index
    index_file = os.path.join(folder, _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant))
    safe_index_file = os.path.join(folder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (_add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant), _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            if is_safetensors_available()
            else (_add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant),)
        )
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True  # load safe due to preference
            else:
                logger.warning(
                    f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
                )
        elif not index_present:
            load_safe = True

    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    # If strict=True, error before loading any of the state dicts.
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    loader = safe_load_file if load_safe else partial(paddlenlp_load, map_location="cpu")

    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        with warnings.catch_warnings():
            warnings.resetwarnings()
            warnings.filterwarnings("ignore", message=r".*is not found in the provided dict.*")
            logger.info(f"set state-dict: {model.set_state_dict(state_dict)}")

        # Make sure memory is fred before we load the next state dict.
        del state_dict
        gc.collect()

    # Return the same thing as PaddlePaddle set_state_dict function.
    return missing_keys, unexpected_keys


def faster_set_state_dict(model, state_dict):
    # the state_dict will be destroied.
    with paddle.no_grad():
        for k, v in model.state_dict().items():
            if k in state_dict:
                v_new = state_dict.pop(k)
                if not isinstance(v_new, paddle.Tensor):
                    raise ValueError(
                        f"faster_set_state_dict need state dict with paddle.Tensor, but got {type(v_new)}"
                    )
                # 2. cast param / Tensor to dtype
                if v.dtype != v_new.dtype:
                    raise ValueError(f"for key: {k}, expect dtype {v.dtype}, but got {v_new.dtype}")
                # check shape
                if list(v.shape) != list(v_new.shape):
                    raise ValueError(f"for key: {k}, expect shape {v.shape}, but got {v_new.shape}")

                dst_tensor = v.value().get_tensor()
                place = v.place

                if not v_new.place._equals(place):
                    # clear dst_tensor for save memory
                    dst_tensor._clear()
                    # v_new = v_new._copy_to(paddle.CUDAPinnedPlace(), False)
                    new_t = v_new._copy_to(place, False)
                else:
                    new_t = v_new

                # 4. share Tensor to origin param / Tensor
                src_tensor = new_t.value().get_tensor()
                dst_tensor._share_data_with(src_tensor)


def _load_state_dict_into_model(model_to_load, state_dict, start_prefix):
    # torch will cast dtype in load_state_dict, but paddle strictly check dtype
    _convert_state_dict_dtype_and_shape(state_dict, model_to_load)

    error_msgs = []

    if len(start_prefix) > 0:
        for key in list(state_dict.keys()):
            if key.startswith(start_prefix):
                state_dict[key.replace(start_prefix, "")] = state_dict.pop(key)

    # TODO: add return status to state_dict
    with warnings.catch_warnings(record=True) as w:
        warnings.resetwarnings()
        # paddlenlp hold  missing_keys , just ignore not found warnings.
        warnings.filterwarnings("ignore", message=r".*is not found in the provided dict.*")
        model_to_load.set_state_dict(state_dict)
        error_msgs.extend([str(x.message) for x in w])

    del state_dict

    return error_msgs


def _convert_state_dict_dtype_and_shape(state_dict, model_to_load):
    # convert the dtype of state dict
    def is_0d_or_1d(tensor):
        return len(tensor.shape) == 0 or list(tensor.shape) == [1]

    for key, value in model_to_load.state_dict().items():
        if key in state_dict:
            if isinstance(state_dict[key], np.ndarray):
                raise ValueError(
                    "convert_state_dict_dtype expected paddle.Tensor not numpy.ndarray, plase convert numpy.ndarray to paddle.Tensor"
                )
            if state_dict[key].is_floating_point() and state_dict[key].dtype != value.dtype:
                state_dict[key] = paddle.cast(state_dict.pop(key), value.dtype)

            # unified 0d and 1d tensor
            if is_0d_or_1d(value) and is_0d_or_1d(state_dict[key]):
                if list(value.shape) != list(state_dict[key].shape):
                    state_dict[key] = paddle.reshape(state_dict.pop(key), value.shape)


def _load_state_dict_into_meta_model(
    model,
    state_dict,
    loaded_state_dict_keys,  # left for now but could be removed, see below
    start_prefix,
    expected_keys,
    dtype=None,
    is_safetensors=False,
    keep_in_fp32_modules=None,
):
    """
    This is somewhat similar to `_load_state_dict_into_model`, but deals with a model that has some or all of its
    params on a `meta` device. It replaces the model params with the data from the `state_dict`, while moving the
    params back to the normal device, but only for `loaded_state_dict_keys`.

    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`

    """
    from paddle.common_ops_import import convert_np_dtype_to_dtype_

    dtype = convert_np_dtype_to_dtype_(dtype)
    error_msgs = []

    for param_name, param in state_dict.items():
        # First part of the test is always true as loaded_state_dict_keys always contains state_dict keys.
        if param_name not in loaded_state_dict_keys or param_name not in expected_keys:
            continue

        if param_name.startswith(start_prefix):
            param_name = param_name[len(start_prefix) :]

        if param.place != paddle.framework._current_expected_place():
            param = param._copy_to(paddle.framework._current_expected_place(), False)

        # # We convert floating dtypes to the `dtype` passed. We want to keep the buffers/params
        # # in int/uint/bool and not cast them.
        if dtype is not None and paddle.is_floating_point(param):
            if (
                keep_in_fp32_modules is not None
                and any(module_to_keep_in_fp32 in param_name for module_to_keep_in_fp32 in keep_in_fp32_modules)
                and dtype == paddle.float16
            ):
                param = param.astype(dtype=paddle.float32)
            else:
                param = param.astype(dtype=dtype)

        if dtype is None:
            old_param = model
            splits = param_name.split(".")
            for split in splits:
                old_param = getattr(old_param, split)
                if old_param is None:
                    break

            if old_param is not None:
                param = param.astype(dtype=old_param.dtype)

        with paddle.no_grad():
            model.state_dict()[param_name].get_tensor()._share_data_with(param.value().get_tensor())
            param.value().get_tensor()._clear()

    return error_msgs


@six.add_metaclass(InitTrackerMeta)
class PretrainedModel(Layer, GenerationMixin, ConversionMixin):
    """
    The base class for all pretrained models. It mainly provides common methods
    for loading (construction and loading) and saving pretrained models. Loading
    and saving also rely on the following class attributes which should be overridden
    by derived classes accordingly:

    - **model_config_file** (str): Represents the file name of model configuration
      for configuration saving and loading in local file system. The value is
      `model_config.json`.
    - **resource_files_names** (dict): Name of local file where the model configuration
      can be saved and loaded locally. Currently, resources only include the model state,
      thus the dict only includes `'model_state'` as key with corresponding
      value `'model_state.pdparams'` for model weights saving and loading.
    - **pretrained_init_configuration** (dict): Provides the model configurations
      of built-in pretrained models (contrasts to models in local file system).
      It has pretrained model names as keys (such as `bert-base-uncased`), and
      the values are dict preserving corresponding configuration for model initialization.
    - **pretrained_resource_files_map** (dict): Provides resource URLs of built-in
      pretrained models (contrasts to models in local file system).
      It has the same key as resource_files_names (that is "model_state"),
      and the corresponding value is a dict with specific model name to model weights URL mapping
      (such as "bert-base-uncased" ->
      "https://bj.bcebos.com/paddlenlp/models/transformers/bert-base-uncased.pdparams").
    - **base_model_prefix** (str): Represents the attribute associated to the
      base model in derived classes of the same architecture adding layers on
      top of the base model. Note: A base model class is pretrained model class
      decorated by `register_base_model`, such as `BertModel`; A derived model
      class is a pretrained model class adding layers on top of the base model,
      and it has a base model as attribute, such as `BertForSequenceClassification`.

    Methods common to models for text generation are defined in `GenerationMixin`
    and also inherited here.

    Besides, metaclass `InitTrackerMeta` is used to create `PretrainedModel`,
    by which subclasses can track arguments for initialization automatically.
    """

    # Deprecated(wj-Mcat): after 2.6.* version
    # save the old-school `LEGACY_CONFIG_NAME`, and will be changed to `CONFIG_NAME` after 2.6.* version
    model_config_file = LEGACY_CONFIG_NAME

    pretrained_init_configuration = {}
    # TODO: more flexible resource handle, namedtuple with fields as:
    # resource_name, saved_file, handle_name_for_load(None for used as __init__
    # arguments), handle_name_for_save
    resource_files_names = {"model_state": PADDLE_WEIGHTS_NAME}
    pretrained_resource_files_map = {}
    base_model_prefix = ""
    main_input_name = "input_ids"
    config_class = None
    _keep_in_fp32_modules = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    _tied_weights_keys = None

    def __init__(self, *args, **kwargs):
        super(PretrainedModel, self).__init__()

        if not self.constructed_from_pretrained_config():
            return

        # extract config from args
        config = None
        for arg in args:
            if isinstance(arg, PretrainedConfig):
                config = arg
                break
        if config is not None:
            self.config: PretrainedConfig = config
            self.model_config_file = CONFIG_NAME
            return

        # extract config from kwargs
        if "config" not in kwargs:
            raise ValueError(
                "PretrainedConfig instance not found in the arguments, you can set it as args or kwargs with config field"
            )

        config = kwargs["config"]
        if not isinstance(config, PretrainedConfig):
            raise TypeError("config parameter should be the instance of PretrainedConfig")

        self.config: PretrainedConfig = kwargs["config"]
        self.model_config_file = CONFIG_NAME
        self.warnings_issued = {}

    def _post_init(self, original_init, *args, **kwargs):
        """
        It would be hooked after `__init__` to add a dict including arguments of
        `__init__` as a attribute named `config` of the pretrained model instance.
        """
        if not self.constructed_from_pretrained_config():
            init_dict = fn_args_to_dict(original_init, *((self,) + args), **kwargs)
            self.config = init_dict

        # only execute when it's the base method
        if (
            original_init.__module__ != "paddlenlp.transformers.model_utils"
            and self.__class__.init_weights is PretrainedModel.init_weights
        ):
            self.init_weights()

    def _init_weights(self, layer):
        """
        Initialize the weights. This method should be overridden by derived class.
        """
        pass

    def _initialize_weights(self, layer):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(layer, "_is_initialized", False):
            return
        self._init_weights(layer)
        layer._is_initialized = True

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # call pure
        if _init_weights:
            # Initialize weights
            self.apply(self._initialize_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways

            # TODO(wj-Mcat): enable all tie-weights later
            # self.tie_weights()

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            dtype (`paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model under this dtype.
        """
        dtype = kwargs.pop("dtype", None)

        if dtype is None:
            if config.dtype is not None:
                dtype = config.dtype
            else:
                dtype = paddle.get_default_dtype()

        with dtype_guard(dtype):
            model = cls(config, **kwargs)

        return model

    @property
    def base_model(self):
        """
        PretrainedModel: The body of the same model architecture. It is the base
            model itself for base model or the base model attribute for derived
            model.
        """
        return getattr(self, self.base_model_prefix, self)

    @property
    def model_name_list(self):
        """
        list: Contains all supported built-in pretrained model names of the
            current PretrainedModel class.
        """
        # Todo: return all model name
        return list(self.pretrained_init_configuration.keys())

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.
        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation
        if "GenerationMixin" in str(self.prepare_inputs_for_generation):
            return False
        return True

    def recompute_enable(self):
        r"""
        Enable Recompute.
        All layers with the `enable_recompute` attribute will be set to `True`
        """

        def fn(layer):
            if hasattr(layer, "enable_recompute") and (layer.enable_recompute is False or layer.enable_recompute == 0):
                layer.enable_recompute = True

        self.apply(fn)

    def recompute_disable(self):
        r"""
        Disable Recompute.
        All layers with the `enable_recompute` attribute will be set to `False`
        """

        def fn(layer):
            if hasattr(layer, "enable_recompute") and (layer.enable_recompute is False or layer.enable_recompute == 0):
                layer.enable_recompute = True

        self.apply(fn)

    def get_memory_footprint(self, return_buffers=True):
        r"""
        Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
        Useful to benchmark the memory footprint of the current model and design some tests.

        Arguments:
            return_buffers (`bool`, *optional*, defaults to `True`):
                Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
                are tensors that do not require gradients and not registered as parameters
        """
        mem = sum([param.numel().item() * param.element_size() for param in self.parameters()])
        if return_buffers:
            mem_bufs = sum([buf.numel().item() * buf.element_size() for buf in self.buffers()])
            mem = mem + mem_bufs
        return mem

    def get_input_embeddings(self) -> nn.Embedding:
        """get input embedding of model

        Returns:
            nn.Embedding: embedding of model
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()

        raise NotImplementedError(
            f"model of {type(base_model)} has not implemented the `get_input_embeddings`"
            " or `set_input_embeddings` method"
        )

    def set_input_embeddings(self, value: Embedding):
        """set new input embedding for model

        Args:
            value (Embedding): the new embedding of model

        Raises:
            NotImplementedError: Model has not implement `set_input_embeddings` method
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.set_input_embeddings(value)
        raise NotImplementedError(
            f"model of {type(base_model)} has not implemented the `get_input_embeddings`"
            " or `set_input_embeddings` method"
        )

    def get_output_embeddings(self) -> Optional[Embedding]:
        """To be overwrited for models with output embeddings

        Returns:
            Optional[Embedding]: the otuput embedding of model
        """
        return None

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        if self.config.tie_word_embeddings:
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if output_embeddings is not None and input_embeddings is not None:
                if input_embeddings.weight.shape != output_embeddings.weight.shape:
                    logger.warning(
                        f"The shape of input embeddings is {input_embeddings.weight.shape} and the shape of output embeddings is {output_embeddings.weight.shape}. "
                        "This is only expected if you are calling the `resize_token_embeddings` method"
                    )
                output_embeddings.weight = input_embeddings.weight
                if getattr(output_embeddings, "bias", None) is not None:
                    # need to pad
                    if output_embeddings.weight.shape[0] > output_embeddings.bias.shape[0]:
                        old_bias = output_embeddings.bias
                        pad_length = output_embeddings.weight.shape[0] - old_bias.shape[0]
                        output_embeddings.bias = output_embeddings.create_parameter(
                            shape=[output_embeddings.weight.shape[0]],
                            attr=output_embeddings._bias_attr,
                            dtype=output_embeddings._dtype,
                            is_bias=True,
                        )
                        new_bias = paddle.concat(
                            [old_bias, paddle.zeros([pad_length], dtype=output_embeddings.bias.dtype)]
                        )
                        output_embeddings.bias.set_value(new_bias)
                    # need to trim
                    elif output_embeddings.weight.shape[0] < output_embeddings.bias.shape[0]:
                        new_bias = output_embeddings.bias[: output_embeddings.weight.shape[0]]
                        output_embeddings.bias = output_embeddings.create_parameter(
                            shape=[output_embeddings.weight.shape[0]],
                            attr=output_embeddings._bias_attr,
                            dtype=output_embeddings._dtype,
                            is_bias=True,
                        )
                        output_embeddings.bias.set_value(new_bias)

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """resize position embedding, this method should be overrited overwrited by downstream models

        Args:
            new_num_position_embeddings (int): the new position size

        Raises:
            NotImplementedError: when called and not be implemented
        """
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `{self.__class__.__module__}.py`"
        )

    @classmethod
    def constructed_from_pretrained_config(cls, init_func=None) -> bool:
        """check if the model is constructed from `PretrainedConfig`
        Returns:
            bool: if the model is constructed from `PretrainedConfig`
        """
        return cls.config_class is not None and issubclass(cls.config_class, PretrainedConfig)

    def save_model_config(self, save_dir: str):
        """
        Deprecated, please use `.config.save_pretrained()` instead.
        Saves model configuration to a file named "config.json" under `save_dir`.

        Args:
            save_dir (str): Directory to save model_config file into.
        """
        logger.warning("The `save_model_config` is deprecated! Please use `.config.save_pretrained()` instead.")
        self.config.save_pretrained(save_dir)

    def save_to_hf_hub(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        subfolder: Optional[str] = None,
        commit_message: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all elements of this model to a new HuggingFace Hub repository.
        Args:
            repo_id (str): Repository name for your model/tokenizer in the Hub.
            private (bool, optional): Whether the model/tokenizer is set to private
            subfolder (str, optional): Push to a subfolder of the repo instead of the root
            commit_message (str, optional) — The summary / title / first line of the generated commit. Defaults to: f"Upload {path_in_repo} with huggingface_hub"
            revision (str, optional) — The git revision to commit from. Defaults to the head of the "main" branch.
            create_pr (boolean, optional) — Whether or not to create a Pull Request with that commit. Defaults to False.
                If revision is not set, PR is opened against the "main" branch. If revision is set and is a branch, PR is opened against this branch.
                If revision is set and is not a branch name (example: a commit oid), an RevisionNotFoundError is returned by the server.

        Returns: The url of the commit of your model in the given repository.
        """
        repo_url = create_repo(repo_id, private=private, exist_ok=True)

        # Infer complete repo_id from repo_url
        # Can be different from the input `repo_id` if repo_owner was implicit
        _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)

        repo_id = f"{repo_owner}/{repo_name}"

        # Check if README file already exist in repo
        try:
            get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename="README.md", revision=revision))
            has_readme = True
        except EntryNotFoundError:
            has_readme = False

        with tempfile.TemporaryDirectory() as root_dir:
            if subfolder is not None:
                save_dir = os.path.join(root_dir, subfolder)
            else:
                save_dir = root_dir
            # save model
            self.save_pretrained(save_dir)
            # Add readme if does not exist
            logger.info("README.md not found, adding the default README.md")
            if not has_readme:
                with open(os.path.join(root_dir, "README.md"), "w") as f:
                    f.write(f"---\nlibrary_name: paddlenlp\n---\n# {repo_id}")

            # Upload model and return
            logger.info(f"Pushing to the {repo_id}. This might take a while")
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=root_dir,
                commit_message=commit_message,
                revision=revision,
                create_pr=create_pr,
            )

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model according to new_num_tokens.

        Args:
            new_num_tokens (Optional[int]):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or None, just
                returns a pointer to the input tokens embedding module of the model without doing anything.

        Returns:
            paddle.nn.Embedding: The input tokens Embeddings Module of the model.
        """
        old_embeddings: nn.Embedding = self.get_input_embeddings()
        if not new_num_tokens or new_num_tokens == old_embeddings.weight.shape[0]:
            return old_embeddings

        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # 2. Update vocab_size
        self.base_model.config["vocab_size"] = new_num_tokens
        self.vocab_size = new_num_tokens

        # update init_config
        self._update_init_config(self.init_config, "vocab_size", new_num_tokens)

        # Tie the weights between the input embeddings and the output embeddings if needed.
        self.tie_weights()

        return new_embeddings

    def _update_init_config(self, init_config: dict, key: str, value: Any):
        """update init_config by <key, value> pair

        Args:
            init_config (dict): the init_config instance
            key (str): the key field
            value (Any): the new value of instance
        """
        if key in init_config:
            init_config[key] = value
            return

        for arg in init_config.get("init_args", []):
            if not isinstance(arg, PretrainedModel):
                continue
            self._update_init_config(arg.init_config, key, value)

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (nn.Embedding):
                Old embeddings to be resized.
            new_num_tokens (Optional[int]):
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end.

        Returns:
            paddle.nn.Embedding: The resized Embedding Module or the old Embedding Module if new_num_tokens is None.
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.shape
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that old_embeddings are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            padding_idx=old_embeddings._padding_idx,
            sparse=old_embeddings._sparse,
        )

        # make sure that new_embeddings's dtype is same as the old embeddings' dtype
        if new_embeddings.weight.dtype != old_embeddings.weight.dtype:
            new_embeddings.to(dtype=old_embeddings.weight.dtype)

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        with paddle.no_grad():
            new_embeddings.weight[:n, :] = old_embeddings.weight[:n, :]

        return new_embeddings

    def __setattr__(self, name, value):
        value = adapt_stale_fwd_patch(self, name, value)
        return super(PretrainedModel, self).__setattr__(name, value)

    @classmethod
    def _resolve_model_file_path(
        cls: Type[PretrainedModel],
        pretrained_model_name_or_path: str,
        from_hf_hub: bool = False,
        cache_dir: str | None = None,
        subfolder: str = "",
        config: PretrainedConfig = None,
        convert_from_torch: bool = False,
        use_safetensors: bool | None = None,
        variant=None,
    ) -> str:

        """resolve model target file path from `` and `cache_dir`

        1. when it is file path:
            return the weight file

        2. when it is model-name:
            2.1 check default `MODEL_HOME` + `model-mame` + model_state.pdparams
            2.2 get the url from `pretrained_resource_files_map`, and set it to `pretrained_model_name_or_path`

        3. when it is local dir:
            check whether the file<local_dir + weight_file> exist

        Args:
            cls (Type[PretrainedModel]): the inherited PretrainedModel class
            pretrained_model_name_or_path (str): the model-name/url/local_dir/local_dir
            cache_dir (Optional[str], optional): cache_dir is used when name_or_path is model-name/url. Defaults to None.
            convert_from_torch (bool, optional): whether support convert pytorch model to paddle model

        Returns:
            str: the model weight file path
        """
        is_sharded = False
        sharded_metadata = None

        # -1. when it's from HF
        if from_hf_hub or convert_from_torch:
            resolved_archive_file = resolve_weight_file_from_hf_hub(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                support_conversion=convert_from_torch,
                subfolder=subfolder,
            )
            return resolved_archive_file, sharded_metadata, is_sharded

        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)

            def get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, variant):
                return os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))

            # pretrained_model_name_or_path is dir
            if is_local:
                if use_safetensors is not False and os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, variant)
                ):
                    # Load from a safetensors checkpoint
                    archive_file = get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, variant)
                elif use_safetensors is not False and os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, weight_name_suffix())
                ):
                    # Load from a safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, weight_name_suffix()
                    )
                elif use_safetensors is not False and os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME, variant)
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME, variant
                    )
                    is_sharded = True
                elif use_safetensors is not False and os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME, weight_name_suffix()
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME, weight_name_suffix()
                    )
                    is_sharded = True
                elif os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_NAME, variant)
                ):
                    # Load from a PaddlePaddle checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_NAME, variant
                    )
                elif os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_INDEX_NAME, variant)
                ):
                    # Load from a sharded PaddlePaddle checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_INDEX_NAME, variant
                    )
                    is_sharded = True
                elif os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_INDEX_NAME, weight_name_suffix()
                    )
                ):
                    # Load from a sharded PaddlePaddle checkpoint for hybrid parallel model
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_INDEX_NAME, weight_name_suffix()
                    )
                    is_sharded = True
                elif os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_NAME,
                        weight_name_suffix(),
                    )
                ):
                    # Load from a PaddlePaddle checkpoint for hybrid parallel model
                    archive_file = get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_NAME,
                        weight_name_suffix(),
                    )
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(PYTORCH_WEIGHTS_NAME, variant))
                ):
                    raise ValueError(
                        f"Found {_add_variant(PYTORCH_WEIGHTS_NAME, variant)} in directory"
                        f" {pretrained_model_name_or_path}. Please set convert_from_torch=True in from_pretrained. eg, Model.from_pretrained(model_name, convert_from_torch=True) "
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(PADDLE_WEIGHTS_NAME, variant)}, found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
            # pretrained_model_name_or_path is file
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = get_path_from_url_with_filelock(pretrained_model_name_or_path)
            else:
                # set correct filename
                if use_safetensors is not False:
                    filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
                else:
                    filename = _add_variant(PADDLE_WEIGHTS_NAME, variant)

                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = dict(
                        cache_dir=cache_dir,
                        subfolder=subfolder,
                        _raise_exceptions_for_missing_entries=False,
                    )
                    resolved_archive_file = None
                    if pretrained_model_name_or_path in cls.pretrained_init_configuration:
                        # fetch the weight url from the `pretrained_resource_files_map`
                        resource_file_url = cls.pretrained_resource_files_map["model_state"][
                            pretrained_model_name_or_path
                        ]
                        resolved_archive_file = cached_file(
                            resource_file_url, _add_variant(PADDLE_WEIGHTS_NAME, variant), **cached_file_kwargs
                        )

                    if resolved_archive_file is None:
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, filename, **cached_file_kwargs
                        )
                    else:
                        # xxx.pdparams in pretrained_resource_files_map renamed model_state.pdparams
                        filename = _add_variant(PADDLE_WEIGHTS_NAME, variant)

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                        elif use_safetensors:
                            raise EnvironmentError(
                                f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} and thus cannot be loaded with `safetensors`. Please make sure that the model has been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                            )
                        else:
                            # This repo has no safetensors file of any kind, we switch to PyTorch.
                            filename = _add_variant(PADDLE_WEIGHTS_NAME, variant)
                            resolved_archive_file = cached_file(
                                pretrained_model_name_or_path, filename, **cached_file_kwargs
                            )
                    if resolved_archive_file is None and filename == _add_variant(PADDLE_WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        # raise ValueError(resolved_archive_file)
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None:
                        # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                        # message.
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(PADDLE_WEIGHTS_NAME, variant)}."
                        )
                except Exception:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://paddlenlp.bj.bcebos.com'"
                    )

            if is_local:
                logger.info(f"Loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(f"Loading weights file {filename} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                subfolder=subfolder,
            )

        return resolved_archive_file, sharded_metadata, is_sharded

    @classmethod
    def _load_pretrained_model(
        cls,
        model: PretrainedModel,
        state_dict: Dict[str, Tensor],
        loaded_keys: List[str],
        resolved_archive_file,
        pretrained_model_name_or_path,
        config=None,
        ignore_mismatched_sizes=False,
        low_cpu_mem_usage=False,
        dtype=None,
        keep_in_fp32_modules=None,
    ) -> Tuple[List[str]]:
        """load the state_dict into model, and do the following things:

            * check the

        Args:
            model (PretrainedModel): the pretrained model instance
            state_dict (Dict[str, Tensor]): the model state dict data
            loaded_keys (List[str]):
            ignore_mismatched_sizes (bool, optional): whether ignore error when tensor size mismatched. Defaults to False.
            dtype (_type_, optional): the dtype of model state dict. Defaults to None.

        Returns:
            Tuple[List[str]]: _description_
        """
        is_safetensors = False

        model_state_dict = model.state_dict()

        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            _prefix = f"{prefix}."
            expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
            expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        # Set some modules to fp32 if any
        if keep_in_fp32_modules is not None:
            for name, param in model.named_parameters():
                if any(module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in keep_in_fp32_modules):
                    param = param.to(dtype=paddle.float32)

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
            base_model_expected_keys = list(model_to_load.state_dict().keys())
            if any(key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys):
                raise ValueError(
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    # If the checkpoint is sharded, we may not have the key here.
                    if checkpoint_key not in state_dict:
                        continue
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{checkpoint_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                        model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            # DONT Hold tensor parallel here, only hold afer load state dict.
            # Whole checkpoint
            # For model parallel if FastGeneration
            # To avoid recursive import temporarily.
            import paddlenlp.ops.fast_transformer.transformer.decoding as ft_decoding

            state_dict = ft_decoding.get_ft_para_conf().fit_partial_model(model_to_load, state_dict)

            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
        else:
            # Sharded checkpoint or whole but low_cpu_mem_usage==True

            # This should always be a list but, just to be sure.
            if not isinstance(resolved_archive_file, list):
                resolved_archive_file = [resolved_archive_file]

            error_msgs = []
            mismatched_keys = []

            if len(resolved_archive_file) > 1:
                resolved_archive_file = tqdm(resolved_archive_file, desc="Loading checkpoint shards")

            for shard_file in resolved_archive_file:
                pre_tensor_parallel_split = False
                if (
                    shard_file.endswith(".safetensors")
                    and config.tensor_parallel_degree > 1
                    and "tp" not in shard_file
                ):
                    pre_tensor_parallel_split = True
                    assert loaded_keys is not None, "loaded_keys is not None."
                    tp_actions = cls.get_tensor_parallel_convert_actions(config, loaded_keys)

                state_dict = load_state_dict(shard_file, tp_actions if pre_tensor_parallel_split else None)

                # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
                # matching the weights in the model.
                mismatched_keys += _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    loaded_keys,
                    add_prefix_to_model,
                    remove_prefix_from_model,
                    ignore_mismatched_sizes,
                )

                if config.tensor_parallel_degree > 1 and ".tp" not in shard_file and not pre_tensor_parallel_split:
                    logger.info("Converting state_dict to Tensor Parallel Format")
                    # ignore error for multi shard, since only parts of data
                    state_dict = cls.convert_tensor_parallel(
                        None, config, state_dict=state_dict, ignore_error=len(resolved_archive_file) > 1
                    )
                    logger.info("Converted state_dict to Tensor Parallel Format")

                if low_cpu_mem_usage:
                    new_error_msgs = _load_state_dict_into_meta_model(
                        model_to_load,
                        state_dict,
                        loaded_keys,
                        start_prefix,
                        expected_keys,
                        dtype=dtype,
                        is_safetensors=is_safetensors,
                        keep_in_fp32_modules=keep_in_fp32_modules,
                    )
                    error_msgs += new_error_msgs
                else:
                    error_msgs += _load_state_dict_into_model(model_to_load, state_dict, start_prefix)

                # force memory release
                del state_dict
                gc.collect()

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if " but the expected shape is" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, from_hf_hub: bool = False, subfolder: str | None = None, *args, **kwargs
    ):
        """
        Creates an instance of `PretrainedModel`. Model weights are loaded
        by specifying name of a built-in pretrained model, a pretrained model from HF Hub, a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of a built-in pretrained model
                - Name of a pretrained model from HF Hub
                - Name of a community-contributed pretrained model.
                - Local directory path which contains model weights file("model_state.pdparams")
                  and model config file ("model_config.json").
            from_hf_hub (bool): load model from huggingface hub. Default to `False`.
            subfolder (str, optional) An optional value corresponding to a folder inside the repo.
                Only works when loading from Huggingface Hub.
            *args (tuple): Position arguments for model `__init__`. If provided,
                use these as position argument values for model initialization.
            **kwargs (dict): Keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for model
                initialization. If the keyword is in `__init__` argument names of
                base model, update argument values of the base model; else update
                argument values of derived model.
            load_state_as_np (bool, optional): The weights read in can be choosed
                to place on CPU or GPU though the model is on the default device.
                If `True`, load the model weights as `numpy.ndarray` on CPU.
                Otherwise, weights would be loaded as tensors on the default
                device. Note that if on GPU, the latter would creates extra
                temporary tensors in addition to the model weights, which
                doubles the memory usage . Thus it is suggested to use `True`
                for big models on GPU. Default to `False`.

        Returns:
            PretrainedModel: An instance of `PretrainedModel`.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertForSequenceClassification

                # Name of built-in pretrained model
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of pretrained model from PaddleHub
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of community-contributed pretrained model
                model = BertForSequenceClassification.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned', num_labels=3)

                # Load from local directory path
                model = BertForSequenceClassification.from_pretrained('./my_bert/')
        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        dtype = kwargs.pop("dtype", None)
        subfolder = kwargs.pop("subfolder", "")
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        convert_from_torch = kwargs.pop("convert_from_torch", None)
        load_state_as_np = kwargs.pop("load_state_as_np", None)
        if load_state_as_np is not None:
            logger.warning("`load_state_as_np` is deprecated,  please delete it!")

        model_kwargs = kwargs

        # from_hf_hub defalut enable convert_from_torch
        if from_hf_hub and convert_from_torch is None:
            logger.warning(
                "If you are attempting to load weights from Hugging Face Hub and want to disable the default behavior of considering torch weights,"
                " you can set ·convert_from_torch=False·. By default, `convert_from_torch` is set to `True`. "
            )
            convert_from_torch = True
        # convert_from_torch defalut is False
        if convert_from_torch is None:
            convert_from_torch = False

        cache_dir = resolve_cache_dir(pretrained_model_name_or_path, from_hf_hub, cache_dir)
        # 1. get the PretrainedConfig to init model
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                from_hf_hub=from_hf_hub,
                subfolder=subfolder,
                **kwargs,
            )
        if not os.path.exists(os.path.join(cache_dir, CONFIG_NAME)):
            config.save_pretrained(cache_dir)

        # refine options for config
        convert_from_torch = cls.support_conversion(config) and convert_from_torch

        if dtype is None:
            dtype = config.dtype
        else:
            config.dtype = dtype

        init_contexts = []
        if low_cpu_mem_usage:
            # Instantiate model.
            init_contexts.append(no_init_weights(_enable=True))
            if is_paddle_support_lazy_init():
                init_contexts.append(paddle.LazyGuard())

        if dtype:
            init_contexts.append(dtype_guard(dtype))

        # Keep in fp32 modules
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False

        # resolve model_weight file
        resolved_archive_file, sharded_metadata, is_sharded = cls._resolve_model_file_path(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            subfolder=subfolder,
            from_hf_hub=from_hf_hub,
            config=config,
            convert_from_torch=convert_from_torch,
            use_safetensors=use_safetensors,
            variant=variant,
        )

        # load pt weights early so that we know which dtype to init the model under
        if not is_sharded and state_dict is None:
            # Time to load the checkpoint
            if resolved_archive_file.endswith(PYTORCH_WEIGHTS_NAME):
                if convert_from_torch:
                    # try to get the name-mapping info
                    logger.info(
                        f"Starting to convert pytorch weight file<{resolved_archive_file}> to "
                        f"paddle weight file<{os.path.join(cache_dir, PADDLE_WEIGHTS_NAME)}> ..."
                    )
                    state_dict = cls.convert(resolved_archive_file, config, cache_dir)
                else:
                    raise ValueError(
                        f"download the {PYTORCH_WEIGHTS_NAME} weight file, but model<{cls}> "
                        "don't support conversion from pytorch weight file to paddle weight file "
                    )
            else:
                # 4. loading non-sharded ckpt from the state dict
                if config.tensor_parallel_degree > 1 and resolved_archive_file.endswith("model_state.pdparams"):
                    state_dict = cls.convert_tensor_parallel(resolved_archive_file, config)
                else:
                    state_dict = load_state_dict(resolved_archive_file)

                logger.info("Loaded weights file from disk, setting weights to model.")

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and dtype == "float16"

        if is_sharded:
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            loaded_state_dict_keys = [k for k in state_dict.keys()]

        if low_cpu_mem_usage:  # or use_keep_in_fp32_modules:
            state_dict = None

        # will only support load paddle.Tensor to model.
        if state_dict is not None:
            for k in list(state_dict.keys()):
                if not isinstance(state_dict[k], paddle.Tensor):
                    with device_guard():
                        state_dict[k] = paddle.Tensor(state_dict.pop(k), zero_copy=True)

        # 3. init the model
        init_args = config["init_args"] or ()
        with ContextManagers(init_contexts):
            model = cls(config, *init_args, **model_kwargs)

        if use_keep_in_fp32_modules:
            # low_cpu_mem_usage = True
            keep_in_fp32_modules = model._keep_in_fp32_modules
        else:
            keep_in_fp32_modules = []

        model, missing_keys, unexpected_keys, mismatched_keys = cls._load_pretrained_model(
            model=model,
            state_dict=state_dict,
            loaded_keys=loaded_state_dict_keys,
            resolved_archive_file=resolved_archive_file,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            low_cpu_mem_usage=low_cpu_mem_usage,
            dtype=dtype,
            keep_in_fp32_modules=keep_in_fp32_modules,
        )

        if paddle.in_dynamic_mode():
            return model

        return model, state_dict

    def save_pretrained(
        self,
        save_dir: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = paddle.save,
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Saves model configuration and related resources (model state) as files
        under `save_dir`. The model configuration would be saved into a file named
        "model_config.json", and model state would be saved into a file
        named "model_state.pdparams".

        The `save_dir` can be used in `from_pretrained` as argument value
        of `pretrained_model_name_or_path` to re-load the trained model.

        Args:
            save_dir (str): Directory to save files into.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertForSequenceClassification

                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
                model.save_pretrained('./trained_model/')
                # reload from save_directory
                model = BertForSequenceClassification.from_pretrained('./trained_model/')
        """

        assert not os.path.isfile(save_dir), "Saving directory ({}) should be a directory, not a file".format(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        merge_tensor_parallel = kwargs.get("merge_tensor_parallel", False)
        config_to_save = kwargs.get("config_to_save", None)
        shard_format = kwargs.get("shard_format", "naive")  # support naive pipeline
        # variant = kwargs.get("variant", None)
        # is_main_process = kwargs.get("is_main_process", True)

        save_directory = save_dir

        if safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)
        # Save model config

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert paddle.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5

        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.dtype = str(dtype).split(".")[1]
        if config_to_save is None:
            config_to_save = copy.deepcopy(model_to_save.config)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()
            if config_to_save.tensor_parallel_degree > 1:
                if merge_tensor_parallel:
                    state_dict = model_to_save.merge_tensor_parallel(state_dict, config_to_save)
                    config_to_save.tensor_parallel_degree = 1
                    if config_to_save.tensor_parallel_rank != 0:
                        logger.info("Saving with merge_tensor_parallel, tensor_parallel_rank > 0 don't need save")
                        return
                    if variant is not None and "tp" in variant:
                        variant = "_".join([x for x in variant.split("_") if "tp" not in x])
                else:
                    variant = weight_name_suffix() if variant is None else variant

        # Attach architecture to the config
        config_to_save.architectures = [model_to_save.__class__.__name__]
        # Save the config
        if is_main_process:
            config_to_save.save_pretrained(save_directory)
            if self.can_generate():
                # to do support generation_config
                pass
                # model_to_save.generation_config.save_pretrained(save_directory)

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict.keys():
                    del state_dict[ignore_key]

        # Shard the model if it is too big.
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else PADDLE_WEIGHTS_NAME
        weights_name = _add_variant(weights_name, variant)

        # Save model
        shards, index = shard_checkpoint(
            state_dict, max_shard_size=max_shard_size, weights_name=weights_name, shard_format=shard_format
        )

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".pdparams", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. paddle_model-00001-of-00005
            filename_no_suffix = filename.replace(".pdparams", "").replace(".safetensors", "")
            reg = re.compile("(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
                and is_main_process
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)

        # Save the model
        for shard_file, shard in shards.items():
            if safe_serialization:
                # At some point we will need to deal better with save_function (used for TPU and other distributed
                # joyfulness), but for now this enough.
                for k in list(shard.keys()):
                    if isinstance(shard[k], paddle.Tensor):
                        shard[k] = shard.pop(k).numpy()
                safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "np"})
            else:
                save_function(shard, os.path.join(save_directory, shard_file))

        if index is None:
            path_to_weights = os.path.join(save_directory, _add_variant(PADDLE_WEIGHTS_NAME, variant))
            logger.info(f"Model weights saved in {path_to_weights}")

        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else PADDLE_WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )
