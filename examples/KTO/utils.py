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

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import paddle
import paddle.distributed as dist


def distribute_gather(data):
    gathered_list = [paddle.zeros_like(data) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_list, data)
    # 0 D tensor
    if gathered_list[0].shape == []:
        gathered_list = [item.unsqueeze(0) for item in gathered_list]
    gathered_data = paddle.concat(gathered_list, axis=0)
    return gathered_data


class PaddlePartialState:
    def __init__(self, **kwargs):
        if dist.is_initialized():
            self.num_processes = paddle.distributed.get_world_size()
            self.process_index = paddle.distributed.get_rank()
            # rank 0 as main process
            self.local_process_index = 0
        else:
            self.num_processes = 1
            self.process_index = 0
            self.local_process_index = 0

    def wait_for_everyone(self):
        if dist.is_initialized():
            paddle.distributed.barrier()

    def _goes_first(self, is_main: bool):
        if not is_main:
            self.wait_for_everyone()

        yield

        if is_main:
            self.wait_for_everyone()

    @property
    def is_local_main_process(self) -> bool:
        "Returns whether the current process is the main process on the local node"
        return self.local_process_index == 0

    @contextmanager
    def local_main_process_first(self):
        yield from self._goes_first(self.is_local_main_process)


def paddle_pad_sequence(sequences, padding_value=0, batch_first=False):
    """Fill sequences(np.ndarray) into a fixed-length matrix."""
    if batch_first is True:
        max_size = sequences[0].shape
        trailing_dims = max_size[1:]
        max_len = max([s.shape[0] for s in sequences])
        # batch_size sequence_length, seq_dim
        out_dims = [len(sequences), max_len] + trailing_dims
        out_tensor = paddle.full(out_dims, padding_value, dtype=sequences[0].dtype)
        for i, tensor in enumerate(sequences):
            length = tensor.shape[0]
            out_tensor[i, :length, ...] = tensor
    return out_tensor


@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [paddle.to_tensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = paddle_pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [paddle.to_tensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [paddle.to_tensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = paddle_pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(axis=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = paddle.to_tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


def disable_dropout_in_model(model: paddle.nn.Layer) -> None:
    for module in model.sublayers():
        if isinstance(module, paddle.nn.Dropout):
            module.p = 0


def pad_to_length(tensor: paddle.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> paddle.Tensor:
    if tensor.shape[dim] >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.shape[dim]
        return paddle.concat(
            [
                tensor,
                pad_value * paddle.ones(pad_size, dtype=tensor.dtype),
            ],
            axis=dim,
        )


def trl_sanitze_kwargs_for_tagging(model, tag_names, kwargs=None):

    if kwargs is not None:
        if "tags" not in kwargs:
            kwargs["tags"] = tag_names
        elif "tags" in kwargs and isinstance(kwargs["tags"], list):
            kwargs["tags"].extend(tag_names)
        elif "tags" in kwargs and isinstance(kwargs["tags"], str):
            tag_names.append(kwargs["tags"])
            kwargs["tags"] = tag_names
    return kwargs
