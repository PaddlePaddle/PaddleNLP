# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union

import numpy as np
import paddle
import pandas as pd
from paddle.io import DataLoader
import paddle.distributed as dist

from ...utils import logger


original_concat = paddle.concat
__all__ = [
    "DataProto",
    "union_tensor_dict",
    "TensorDict",
]


class TensorDict:
    def __init__(self, source: dict, batch_size=None, num_batch_dims=1):
        self._tensors = {}
        self.batch_size = batch_size
        self.num_batch_dims = num_batch_dims

        for key, tensor in source.items():
            self[key] = tensor

    def __setitem__(self, key: str, tensor: paddle.Tensor):
        if self.batch_size is not None:
            assert (
                tensor.shape[: self.num_batch_dims] == self.batch_size
            ), f"Batch dim mismatch! Expected {self.batch_size}, got {tensor.shape}"
        self._tensors[key] = tensor

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._tensors[key]
        elif isinstance(key, slice):
            strides = [1] if key.step is None else [key.step]
            tensor_dict_slice = {
                k: paddle.strided_slice(v, axes=[0], starts=[key.start], ends=[key.stop], strides=strides)
                for k, v in self._tensors.items()
            }
            batch_size = tensor_dict_slice[list(tensor_dict_slice.keys())[0]].shape[: self.num_batch_dims]
            return TensorDict(tensor_dict_slice, batch_size=batch_size, num_batch_dims=self.num_batch_dims)
        else:
            raise KeyError(f"Unsupported key type: {type(key)}")

    def __contains__(self, key):
        return key in self._tensors

    def update(self, other):
        if isinstance(other, TensorDict):
            other = other._tensors
        for key, value in other.items():
            # consistency was not checked
            self._tensors[key] = value

    def keys(self):
        return self._tensors.keys()

    def items(self):
        return self._tensors.items()

    def to(self, device: str):
        for key in self._tensors:
            self._tensors[key] = self._tensors[key].to(device)
        return self

    def pop(self, key: str, default=None):
        if key in self._tensors:
            value = self._tensors.pop(key)
            return value
        elif default is not None:
            return default
        else:
            raise KeyError(f"Key '{key}' not found in TensorDict and no default value provided.")

    def select(self, *keys, default=None):
        selected_tensors = {}
        for key in keys:
            if key in self._tensors:
                selected_tensors[key] = self._tensors[key]
            elif default is not None:
                selected_tensors[key] = default
            else:
                raise KeyError(f"Key '{key}' not found in TensorDict and no default value provided.")

        batch_size = (
            list(selected_tensors[list(selected_tensors.keys())[0]].shape[: self.num_batch_dims])
            if selected_tensors
            else None
        )
        return TensorDict(selected_tensors, batch_size=batch_size, num_batch_dims=self.num_batch_dims)

    def rename_key_(self, old_keys, new_keys):
        if isinstance(old_keys, (str,)):
            old_keys = [old_keys]
        if isinstance(new_keys, (str,)):
            new_keys = [new_keys]
        if len(old_keys) != len(new_keys):
            raise ValueError("old_keys and new_keys must have the same length.")

        for old_key, new_key in zip(old_keys, new_keys):
            if old_key not in self._tensors:
                raise KeyError(f"Key '{old_key}' not found in TensorDict.")
            if new_key in self._tensors:
                raise KeyError(f"Key '{new_key}' already exists in TensorDict.")
            self._tensors[new_key] = self._tensors.pop(old_key)
        return self

    @classmethod
    def concat(cls, tensordict_list, axis=0):
        if not tensordict_list:
            raise ValueError("tensordict_list must not be empty")

        first_tensordict = tensordict_list[0]
        first_keys = first_tensordict.keys()
        first_shapes = {key: tensor.shape for key, tensor in first_tensordict.items()}

        for tensordict in tensordict_list:
            if tensordict.keys() != first_keys:
                raise ValueError("All TensorDict objects must have the same keys")
            for key in first_keys:
                if (
                    tensordict[key].shape[:axis] + tensordict[key].shape[axis + 1 :]
                    != first_shapes[key][:axis] + first_shapes[key][axis + 1 :]
                ):
                    raise ValueError(f"Shapes of tensor '{key}' do not match except on concatenation axis {axis}")

        concatenated_tensors = {
            key: paddle.concat([tensordict[key] for tensordict in tensordict_list], axis=axis) for key in first_keys
        }

        batch_size = concatenated_tensors[list(concatenated_tensors.keys())[0]].shape[
            : tensordict_list[0].num_batch_dims
        ]
        return cls(concatenated_tensors, batch_size=batch_size, num_batch_dims=tensordict_list[0].num_batch_dims)
    
    def get(self, key, default=None):
        return self._tensors.get(key, default)


def tensordict_concat(
    x: Union[Sequence[paddle.Tensor], Sequence[TensorDict]],
    axis: int | paddle.Tensor = 0,
    name: str | None = None,
):
    def is_tensor_sequence():
        if isinstance(x[0], paddle.Tensor):
            return True
        else:
            return False

    if not is_tensor_sequence() and paddle.in_dynamic_mode():
        return TensorDict.concat(x)
    else:
        return original_concat(x, axis, name)


paddle.concat = tensordict_concat


def union_two_dict(dict1: Dict, dict2: Dict):
    """Union two dict. Will throw an error if there is an item not the same object with the same key.

    Args:
        dict1:
        dict2:

    Returns:

    """
    for key, val in dict2.items():
        if key in dict1:
            assert dict2[key] == dict1[key], f"{key} in meta_dict1 and meta_dict2 are not the same object"
        dict1[key] = val

    return dict1


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    """Pad a DataProto to size divisible by size_divisor

    Args:
        size_divisor (int): size divisor

    Returns:
        data: (DataProto): the padded DataProto
        pad_size (int)
    """
    assert isinstance(data, DataProto), "data must be a DataProto"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad_dataproto(data: "DataProto", pad_size):
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two tensordicts."""
    assert (
        tensor_dict1.batch_size == tensor_dict2.batch_size
    ), f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            assert tensor_dict1[key].equal(
                tensor_dict2[key]
            ), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"

    return tensor_dict1


def union_numpy_dict(tensor_dict1: dict[np.ndarray], tensor_dict2: dict[np.ndarray]) -> dict[np.ndarray]:
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            # to properly deal with nan and object type
            assert pd.DataFrame(tensor_dict2[key]).equals(
                pd.DataFrame(tensor_dict1[key])
            ), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
        tensor_dict1[key] = val

    return tensor_dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            output[key].append(item)
    return output


def fold_batch_dim(data: "DataProto", new_batch_size):
    """
    Fold a batch dim from [bsz, xxx] into [new_bsz, bsz // new_bsz, xxx]
    """
    batch_size = data.batch.batch_size[0]

    assert batch_size % new_batch_size == 0

    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch

    tensor = tensor.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)

    for key, val in non_tensor.items():
        non_tensor[key] = np.reshape(val, newshape=(new_batch_size, -1, *val.shape[1:]))

    return DataProto(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


def unfold_batch_dim(data: "DataProto", batch_dims=2):
    """
    Unfold the first n dims as new batch dim
    """
    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch
    tensor.auto_batch_size_(batch_dims=batch_dims)
    tensor = tensor.view(-1)

    batch_size = tensor.batch_size[0]

    non_tensor_new = {}

    for key, val in non_tensor.items():
        non_tensor_new[key] = np.reshape(val, newshape=(batch_size, *val.shape[batch_dims:]))

    return DataProto(batch=tensor, non_tensor_batch=non_tensor_new, meta_info=data.meta_info)


def collate_fn(x: list["DataProtoItem"]):
    batch = []
    non_tensor_batch = []
    for data in x:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    batch = paddle.stack(batch)
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.array(val, dtype=object)
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

def make_eos_mask(response_id, eos_token_ids=0, dtype=paddle.int64):
    """
    end of sentence token can be int or list: 1 or [1, 2]
    e.g. eos_token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    """
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    eos_mask = paddle.zeros_like(response_id, dtype=paddle.bool)
    for token_id in eos_token_ids:
        eos_mask |= response_id == token_id

    eos_mask = eos_mask.to("int64")
    eos_mask = (paddle.cumsum(eos_mask, axis=1) - eos_mask).to("bool")
    eos_mask = paddle.logical_not(eos_mask).to(dtype)
    return eos_mask


@dataclass
class DataProtoItem:
    # TODO(zhangchi.usc1992) add consistency check
    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)


@dataclass
class DataProto:
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    """

    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        # perform necessary checking
        self.check_consistency()

    def __len__(self):
        if self.batch is not None:
            return self.batch.batch_size[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            random_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[random_key].shape[0]
        else:
            return 0

    def __getitem__(self, item):
        """
        Enhanced indexing for DataProto objects.

        Args:
            item: Can be one of:
                - int: A single index
                - slice: A slice object (start:stop:step)
                - list: A list of indices
                - numpy.ndarray: An array of indices
                - torch.Tensor: A tensor of indices

        Returns:
            DataProto: For all indexing types except single integers
            DataProtoItem: Only for single integer indices
        """
        # Case 1: Slice object - use the slice method
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)

        # Case 2: List, numpy array, or torch tensor - use sel_idxs
        elif isinstance(item, (list, np.ndarray, paddle.Tensor)):
            return self.select_idxs(item)

        # Case 3: Single integer - return DataProtoItem for backward compatibility
        elif isinstance(item, (int, np.integer)):
            tensor_data = self.batch[item]
            non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
            return_type = DataProto if isinstance(item, slice) else DataProtoItem
            return return_type(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

        # Case 4: Unsupported type
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    def print_size(self, prefix=""):
        size_of_tensordict = 0
        for key, tensor in self.batch.items():
            size_of_tensordict += tensor.element_size() * tensor.numel()
        size_of_numpy_array = 0
        for key, numpy_array in self.non_tensor_batch.items():
            size_of_numpy_array += numpy_array.nbytes

        size_of_numpy_array /= 1024**3
        size_of_tensordict /= 1024**3

        message = f"Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB"

        if prefix:
            message = f"{prefix}, " + message
        print(message)
    def check_consistency(self):
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        if self.non_tensor_batch is not None:
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray)

        if self.batch is not None and len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                assert (
                    isinstance(val, np.ndarray) and val.dtype == object
                ), "data in the non_tensor_batch must be a numpy.array with dtype=object"
                assert (
                    val.shape[0] == batch_size
                ), f"key {key} length {len(val)} is not equal to batch size {batch_size}"

    @classmethod
    def from_single_dict(cls, data: Dict[str, Union[paddle.Tensor, np.ndarray]], meta_info=None):
        tensors = {}
        non_tensors = {}

        for key, val in data.items():
            if isinstance(val, paddle.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @classmethod
    def from_dict(cls, tensors: Dict[str, paddle.Tensor], non_tensors=None, meta_info=None, num_batch_dims=1):
        """Create a DataProto from a dict of tensors. This assumes that
        1. All the tensor in tensors have the same dim0
        2. Only dim0 is the batch dim
        """
        assert len(tensors) > 0, "tensors must not be empty"
        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert (
                    batch_size == current_batch
                ), f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}"

        for key, val in non_tensors.items():
            non_tensors[key] = np.array(val, dtype=object)

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size, num_batch_dims=num_batch_dims)
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)
    
    @staticmethod
    def gather_tensor(tensor, dp_group=None, sd_group=None):
        from ...utils.nested import flatten_list
        """Gather tensor from all devices."""

        if not isinstance(tensor, list):
            tensor = [tensor]

        if isinstance(tensor[0], paddle.Tensor):
            type = "tensor"
        elif isinstance(tensor[0], np.ndarray):
            type = "numpy"
        else:
            raise TypeError(f"{type(tensor[0])} is not supported for gather and pad")

        dtype = tensor[0].dtype

        if (dp_group is None and sd_group is None) or (dp_group.nranks == 1 and sd_group.nranks == 1):
            return tensor
            
        def map_func(weight):
            if isinstance(weight, paddle.Tensor):
                weight = weight.numpy()
            return weight

        tensor = [map_func(i) for i in tensor]

        sd_gathered_tensor = []
        if sd_group.nranks > 1:
            dist.all_gather_object(sd_gathered_tensor, tensor, group=sd_group)

        dp_gathered_tensor = []
        if dp_group.nranks > 1:
            if len(sd_gathered_tensor) > 0:
                tensor = sd_gathered_tensor
            dist.all_gather_object(dp_gathered_tensor, tensor, group=dp_group)

        if len(dp_gathered_tensor) > 0:
            gathered_tensor = dp_gathered_tensor
        else:
            gathered_tensor = sd_gathered_tensor

        if type == "tensor":
            gathered_tensor = [paddle.to_tensor(i, dtype=dtype) for i in flatten_list(gathered_tensor)]

        return gathered_tensor

    @staticmethod
    def concatenate_tensors(
        gathered_list: List[Union[paddle.Tensor, np.ndarray, List]],
        data_parallel_group=None,
        sharding_parallel_group=None
    ) -> Union[paddle.Tensor, np.ndarray]:
        """
        Concatenates a list of tensors/arrays that have been gathered from
        different distributed ranks. Handles both Paddle Tensors and NumPy arrays,
        and accounts for nested lists from `all_gather_object` with multiple ranks.

        Args:
            gathered_list (List[Union[paddle.Tensor, np.ndarray, List]]):
                A list containing tensors or arrays gathered from all ranks.
                This list might be nested if `all_gather_object` was used with
                multiple groups (e.g., sharding then data parallel).
            data_parallel_group (Optional[paddle.distributed.collective.Group]):
                The data parallel communication group.
            sharding_parallel_group (Optional[paddle.distributed.collective.Group]):
                The sharding parallel communication group.

        Returns:
            Union[paddle.Tensor, np.ndarray]: The concatenated tensor/array.
                                              If the input was Paddle Tensor, returns Paddle Tensor.
                                              If the input was NumPy array, returns NumPy array.
        """
        from ...utils.nested import flatten_list

        if not gathered_list:
            raise ValueError("Input gathered_list cannot be empty.")
        is_paddle_tensor = isinstance(gathered_list[0], paddle.Tensor)
        if is_paddle_tensor:
            return paddle.concat(gathered_list, axis=0)
        else:
            return np.concatenate(flatten_list(gathered_list), axis=0)

    # 这个后面要改，从meta_info获取 pad_index
    @staticmethod
    def pad_tensor(tensor_list, pad_index, dtype, padding_side):
        # pad = False if len(tensor_list[0].shape) == 1 else True
        # pad_index = tokenizer.pad_token_id
        # padding_side = "left" if (key == "prompt" or key == "label_ids") else "right"

        # dtype = tensor_list[0].dtype

        max_size = max([i.shape[-1] for i in tensor_list])
        data_num = sum([i.shape[0] for i in tensor_list])
        if isinstance(tensor_list[0], paddle.Tensor):
            new_tensor = paddle.full((data_num, max_size), pad_index, dtype=dtype)
        elif isinstance(tensor_list[0], np.ndarray):
            new_tensor = np.full((data_num, max_size), pad_index, dtype=dtype)

        offset = 0
        for idx, i in enumerate(tensor_list):
            # new_tensor[offset : offset + i.shape[0], : i.shape[-1]] = i
            data_length = i.shape[-1]

            if padding_side == "right":
                new_tensor[offset : offset + i.shape[0], :data_length] = i
            elif padding_side == "left":
                new_tensor[offset : offset + i.shape[0], -data_length:] = i
            else:
                raise ValueError("padding_side must be 'right' or 'left'")
            offset += i.shape[0]
        return new_tensor
    
    @staticmethod
    def pad_or_concat_tensor_list(tensor_list, pad_index, key):
        from ...utils.nested import flatten_list
        left_padding_key = ("prompt", "label_ids")
        # 由于在 gather 后判断是否 pad，所以要用到 flatten_list
        pad = False if len(flatten_list(tensor_list)[0].shape) == 1 else True
        # pad = False if len(tensor_list[0].shape) == 1 else True
        padding_side = "left" if (key in left_padding_key) else "right"
        dtype = flatten_list(tensor_list)[0].dtype
        if not pad:
            return DataProto.concatenate_tensors(tensor_list)
        else:
            return DataProto.pad_tensor(tensor_list, pad_index, dtype, padding_side)

    @staticmethod
    def pad_batch_data(
        tensor_list: List[paddle.Tensor] = None,
        pad_token_id = None
    ) -> List[paddle.Tensor]:
        
        tensor_list = [paddle.unsqueeze(v, axis=0) if v.ndim == 1 else v for v in tensor_list]
        padded_tensors = DataProto.pad_tensor(
            tensor_list,
            pad_index=pad_token_id,
            dtype=tensor_list[0].dtype,
            padding_side="right",
        )
        return padded_tensors

    def to(self, device) -> "DataProto":
        """move the batch to device

        Args:
            device (paddle.device, str): paddle device

        Returns:
            DataProto: the current DataProto

        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "DataProto":
        """Select a subset of the DataProto via batch_keys and meta_info_keys

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to select
            meta_info_keys (list, optional): a list of keys indicating the meta info to select

        Returns:
            DataProto: the DataProto with the selected batch_keys and meta_info_keys
        """
        # TODO (zhangchi.usc1992) whether to copy
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return DataProto(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def select_idxs(self, idxs):
        """
        Select specific indices from the DataProto.

        Args:
            idxs (torch.Tensor or numpy.ndarray or list): Indices to select

        Returns:
            DataProto: A new DataProto containing only the selected indices
        """
        if isinstance(idxs, list):
            idxs = paddle.tensor(idxs, dtype=paddle.int32)

        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_paddle = paddle.from_numpy(idxs)
        else:  # torch.Tensor
            idxs_paddle = idxs
            idxs_np = idxs.numpy()

        if self.batch is not None:
            # Use TensorDict's built-in indexing capabilities
            selected_batch = TensorDict(
                source={key: tensor[idxs_paddle] for key, tensor in self.batch.items()},
                batch_size=(idxs_paddle.shape[0],),
            )
        else:
            selected_batch = None

        selected_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            selected_non_tensor[key] = val[idxs_np]

        return DataProto(batch=selected_batch, non_tensor_batch=selected_non_tensor, meta_info=self.meta_info)

    def slice(self, start=None, end=None, step=None):
        """
        Slice the DataProto and return a new DataProto object.
        This is an improved version of direct slicing which returns a DataProtoItem.

        Args:
            start (int, optional): Start index. Defaults to None (start from beginning).
            end (int, optional): End index (exclusive). Defaults to None (go to end).
            step (int, optional): Step size. Defaults to None (step=1).

        Returns:
            DataProto: A new DataProto containing the sliced data

        Examples:
            # Using the slice method directly
            sliced_data = data_proto.slice(10, 20)

            # Using enhanced indexing (returns DataProto)
            sliced_data = data_proto[10:20]
            sliced_data = data_proto[::2]  # Every other element

            # Using list indexing (returns DataProto)
            indices = [1, 5, 10]
            selected_data = data_proto[indices]

            # Single index still returns DataProtoItem
            single_item = data_proto[5]
        """
        # Create a slice object
        slice_obj = slice(start, end, step)

        # Handle the batch data
        if self.batch is not None:
            # Use TensorDict's built-in slicing capabilities
            sliced_batch = self.batch[slice_obj]
        else:
            sliced_batch = None

        # Handle the non-tensor batch data
        sliced_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            sliced_non_tensor[key] = val[slice_obj]

        # Return a new DataProto object
        return DataProto(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, meta_info=self.meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to pop
            meta_info_keys (list, optional): a list of keys indicating the meta info to pop

        Returns:
            DataProto: the DataProto with the popped batch_keys and meta_info_keys
        """
        assert batch_keys is not None
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []

        tensors = {}
        # tensor batch
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        # non tensor batch
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def rename(self, old_keys=None, new_keys=None) -> "DataProto":
        """
        Note that this function only rename the key in the batch
        """

        def validate_input(keys):
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
            return keys

        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(
                f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}"
            )

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: "DataProto") -> "DataProto":
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if

        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        Args:
            other (DataProto): another DataProto to union

        Returns:
            DataProto: the DataProto after union
        """
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        """Make an iterator from the DataProto.

        Args:
            mini_batch_size (int): mini-batch size when iterating the dataset. We require that
                ``batch.batch_size[0] % mini_batch_size == 0``
            epochs (int): number of epochs when iterating the dataset.
            dataloader_kwargs: internally, it returns a DataLoader over the batch.
                The dataloader_kwargs is the kwargs passed to the DataLoader

        Returns:
            Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration steps is
            ``self.batch.batch_size * epochs // mini_batch_size``
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        assert isinstance(dataloader_kwargs, Dict)

        train_dataloader = DataLoader(
            dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, **dataloader_kwargs
        )

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def chunk(self, chunks: int) -> List["DataProto"]:
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.

        Args:
            chunks (int): the number of chunks to split on dim=0

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        assert (
            len(self) % chunks == 0
        ), f"only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}."

        if self.batch is not None:
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            non_tensor_lst = np.array_split(val, chunks)
            assert len(non_tensor_lst) == chunks
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        output = []
        for i in range(chunks):
            output.append(
                DataProto(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info)
            )

        return output

    @staticmethod
    def concat(data: List["DataProto"]) -> "DataProto":
        """Concat a list of DataProto. The batch is concatenated among dim=0.
        The meta_info is assumed to be identical and will use the first one.

        Args:
            data (List[DataProto]): list of DataProto

        Returns:
            DataProto: concatenated DataProto
        """
        batch_lst = []
        for batch in data:
            batch_lst.append(batch.batch)
        if batch_lst[0] is not None:
            new_batch = paddle.concat(batch_lst, axis=0)
        else:
            new_batch = None

        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        return DataProto(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=data[0].meta_info)

    def reorder(self, indices):
        """
        Note that this operation is in-place
        """
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    def repeat(self, repeat_times=2, interleave=True):
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if self.batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: paddle.repeat_interleave(tensor, repeats=repeat_times, axis=0)
                    for key, tensor in self.batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=[self.batch.batch_size[0] * repeat_times],
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        return DataProto(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )
    
    @staticmethod
    def process_row(row, remove_value=0, remove_side="both", eos_token_id=None):
        """
        Remove leading/trailing specific values from a tensor.

        Args:
            row (paddle.Tensor): The 1D tensor to be processed.
            remove_value (int, optional): The value to be removed, default is 0.
            remove_side (str, optional): The side to remove values from, can be "left" (remove leading only), "right" (remove trailing only),
                or "both" (remove both leading and trailing), default is "both".

        Returns:
            paddle.Tensor: The processed 1D tensor.
        """
        if eos_token_id is not None and remove_value == eos_token_id:
            # Special processing: Retain the index of the last eos_token_id
            is_not_remove_value = row != remove_value
            last_eos_idx = paddle.nonzero(row == eos_token_id).flatten()
            if last_eos_idx.shape[0] > 0:
                last_eos_idx = last_eos_idx[-1]
                is_not_remove_value[last_eos_idx] = True
        else:
            is_not_remove_value = row != remove_value

        non_zero_indices = paddle.nonzero(is_not_remove_value).flatten()
        if non_zero_indices.shape[0] == 0:
            # If the row is all zeros, log a warning and return the original row.
            logger.warning("Row is all zeros, no trimming will be performed.")
            return row
        start_index = non_zero_indices[0]
        end_index = non_zero_indices[-1]
        # Slice the middle non-zero part.
        if remove_side == "left":
            trimmed_row = row[start_index:]
        elif remove_side == "right":
            trimmed_row = row[: end_index + 1]
        elif remove_side == "both":
            trimmed_row = row[start_index : end_index + 1]
        else:
            # If an unknown remove_side is provided, log a warning and use "both".
            logger.warning("Unknown remove_side, using 'both' remove_side.")
            trimmed_row = row[start_index : end_index + 1]

        return trimmed_row
    
    def remove_pad_tokens_after_generate(self, tokenizer, args):
        cleanup_batch = [
                DataProto.process_row(
                    row,
                    remove_value=tokenizer.pad_token_id,
                    remove_side="right",
                    eos_token_id=tokenizer.eos_token_id,
                )
                for row in self.batch["input_ids"]
            ]
        if args.use_rm_server:
            label_ids_batch = [
                    DataProto.process_row(
                        row,
                        remove_value=tokenizer.pad_token_id,
                        remove_side="left",
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    for row in self.batch["label_ids"]
                ]
        index = self.non_tensor_batch["index"]

        return cleanup_batch, index, label_ids_batch

    def split_batch_into_micro_batches(self, batch_size, pad_token_id=0) -> List['DataProto']:
        """
        Splits total_batch into micro-batches of size `batch_size`.

        Args:
            total_batch (dict): Dictionary containing full batched tensors.
            batch_size (int): Micro batch size per device.

        Returns:
            list of dict: A list of micro-batches.
        """
        micro_batches = []
        num_micro_batches = self.batch["input_ids"].shape[0] // batch_size
        if self.batch["input_ids"].shape[0] % batch_size != 0:
            num_micro_batches += 1
        if num_micro_batches <= 0:
            logger.warning(
                "The total batch size is smaller than the batch size, please consider using a smaller batch size or a larger global_batch_size."
            )
            num_micro_batches = 1

        for i in range(num_micro_batches):
            micro_batch = {}
            for key, data in self.batch.items():
                if isinstance(data, paddle.Tensor):
                    micro_batch[key] = data[i * batch_size : (i + 1) * batch_size]
                elif isinstance(data, np.ndarray):
                    micro_batch[key] = data[i * batch_size : (i + 1) * batch_size]
                elif isinstance(data, list):
                    micro_batch[key] = data[i * batch_size : (i + 1) * batch_size]
                else:
                    raise TypeError(f"Unsupported data type for key {key}: {type(data)}")

            # if os.getenv("PROCESS_PROMPT_AND_RESPONSE", "1").lower() in ["1", "t", "true", "yes", "y"]:
            #     micro_batch = process_prompt_and_response(micro_batch=micro_batch, pad_token_id=pad_token_id)

            micro_batches.append(DataProto.from_single_dict(micro_batch))

        return micro_batches