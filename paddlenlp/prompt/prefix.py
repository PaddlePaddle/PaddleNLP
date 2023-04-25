# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

import paddle
import paddle.nn as nn
from paddle.distributed import fleet

from ..utils.env import PREFIX_CONFIG_NAME
from ..utils.log import logger
from .prompt_utils import signature

__all__ = [
    "PrefixConfig",
    "PrefixModelForCausalLM",
]


@dataclass
class PrefixConfig:
    trainable_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to train when applying with Prefix Tuning."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    prefix_dropout: float = field(default=0.0, metadata={"help": "Prefix projection dropout"})
    num_prefix_tokens: Optional[int] = field(default=None, metadata={"help": "Number of prefix tokens"})
    num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
    num_hidden_layers: Optional[int] = field(default=None, metadata={"help": "Number of transformer hidden layers"})
    hidden_size: Optional[int] = field(
        default=None, metadata={"help": "The hidden embedding dimension of the transformer model"}
    )
    prefix_projection: bool = field(default=False, metadata={"help": "Whether to project the prefix tokens"})
    prefix_projection_hidden_size: Optional[int] = field(
        default=None, metadata={"help": "The hidden embedding dimension of the transformer model"}
    )
    tensor_parallel_degree: int = field(default=1, metadata={"help": ("1 for not use tensor parallel")})
    dtype: Optional[str] = field(default=None, metadata={"help": "The data type of tensor"})

    @property
    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return self.__dict__

    def save_pretrained(self, save_directory):
        r"""
        This method saves the configuration of your adapter model in a directory.
        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        output_dict = self.__dict__
        output_path = os.path.join(save_directory, PREFIX_CONFIG_NAME)

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a directory.
        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the hub-id where the configuration is saved.
            **kwargs:
                Additional keyword arguments passed along to the child class initialization.
        """
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, PREFIX_CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, PREFIX_CONFIG_NAME)
        else:
            raise ValueError(f"Can't find prefix_config.json at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)

        config = cls(**kwargs)

        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @classmethod
    def from_json_file(cls, path_json_file):
        r"""
        Loads a configuration file from a json file.
        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object


class PrefixModelForCausalLM(paddle.nn.Layer):
    """
    PrefixModel for causal language modeling.
    """

    def __init__(
        self,
        model,
        prefix_config: PrefixConfig,
        postprocess_past_key_value: Optional[Callable] = None,
        pad_attention_mask: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.prefix_config = prefix_config
        self.model = model
        self.forward_keys = signature(self.model.forward)
        self.config = model.config
        self.prefix_encoder = self._create_prefix_encoder()
        self.prefix_dropout = nn.Dropout(p=prefix_config.prefix_dropout)
        self.prefix_tokens = paddle.arange(self.prefix_config.num_prefix_tokens, dtype="int64")
        self.model_prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
        self.inference = False
        self.postprocess_past_key_value = postprocess_past_key_value
        self.pad_attention_mask = pad_attention_mask

    def forward(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):

        batch_size = input_ids.shape[0]
        kwargs["use_cache"] = True
        past_key_values = self._get_past_key_values(batch_size)

        if attention_mask is not None:

            if self.pad_attention_mask is not None:
                attention_mask = self.pad_attention_mask(
                    input_ids.shape, self.prefix_config.num_prefix_tokens, attention_mask
                )
            else:
                prefix_attention_mask = paddle.ones([batch_size, self.prefix_config.num_prefix_tokens])
                attention_mask = paddle.concat((prefix_attention_mask, attention_mask), axis=1)
            kwargs["attention_mask"] = attention_mask

        if "past_key_values" in self.forward_keys:
            output = self.model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        elif "cache" in self.forward_keys:
            output = self.model(input_ids=input_ids, cache=past_key_values, **kwargs)
        else:
            raise NotImplementedError("Model does not support past_key_values either cache")
        return output

    def generate(self, **kwargs):
        if "input_ids" not in kwargs:
            raise ValueError("input_ids must be provided for Peft model generation")

        self.model.prepare_inputs_for_generation = self._prepare_inputs_for_generation
        outputs = self.model.generate(**kwargs)
        self.model.prepare_inputs_for_generation = self.model_prepare_inputs_for_generation
        return outputs

    def _prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.model_prepare_inputs_for_generation(*args, **kwargs)
        attention_mask = model_kwargs["attention_mask"]
        if self.pad_attention_mask is not None:
            attention_mask = self.pad_attention_mask(
                model_kwargs["input_ids"].shape, self.prefix_config.num_prefix_tokens, attention_mask
            )
        else:
            prefix_attention_mask = paddle.ones(
                [model_kwargs["input_ids"].shape[0], self.prefix_config.num_prefix_tokens]
            )
            attention_mask = paddle.concat((prefix_attention_mask, attention_mask), axis=1)
        model_kwargs["attention_mask"] = attention_mask

        if "past_key_values" in self.forward_keys:
            key = "past_key_values"
        elif "cache" in self.forward_keys:
            key = "cache"
        else:
            raise NotImplementedError("Model does not support past_key_values either cache")
        if model_kwargs[key] is None:
            batch_size = model_kwargs["input_ids"].shape[0]
            past_key_values = self._get_past_key_values(batch_size)
            model_kwargs[key] = past_key_values
        return model_kwargs

    def mark_only_prefix_as_trainable(self) -> None:
        for name, weight in self.model.state_dict().items():
            if self.prefix_config.trainable_modules is not None and any(
                re.fullmatch(trainable_module, name) for trainable_module in self.lora_config.trainable_modules
            ):
                weight.stop_gradient = False
            else:
                weight.stop_gradient = True

        for name, weight in self.prefix_encoder.state_dict().items():
            weight.stop_gradient = False

    def _create_prefix_encoder(self):
        prefix_dropout = nn.Dropout(p=self.prefix_config.prefix_dropout)
        if self.prefix_config.prefix_projection:
            activation = nn.Tanh()
            if self.config.tensor_parallel_degree > 1:
                prefix_embedding = fleet.meta_parallel.VocabParallelEmbedding(
                    self.prefix_config.num_prefix_tokens,
                    self.prefix_config.hidden_size,
                )
                prefix_proj_0 = fleet.meta_parallel.ColumnParallelLinear(
                    self.prefix_config.hidden_size,
                    self.prefix_config.prefix_projection_hidden_size,
                    has_bias=True,
                    gather_output=False,
                )
                prefix_proj_1 = fleet.meta_parallel.RowParallelLinear(
                    self.prefix_config.prefix_projection_hidden_size,
                    self.prefix_config.hidden_size * self.prefix_config.num_hidden_layers * 2,
                    has_bias=True,
                    input_is_parallel=True,
                )
            else:
                prefix_embedding = nn.Embedding(
                    self.prefix_config.num_prefix_tokens,
                    self.prefix_config.hidden_size,
                )
                prefix_proj_0 = nn.Linear(
                    self.prefix_config.hidden_size,
                    self.prefix_config.prefix_projection_hidden_size,
                )
                prefix_proj_1 = nn.Linear(
                    self.prefix_config.prefix_projection_hidden_size,
                    self.prefix_config.hidden_size * self.prefix_config.num_hidden_layers * 2,
                )
            prefix_encoder = nn.Sequential(prefix_embedding, prefix_proj_0, activation, prefix_proj_1, prefix_dropout)
        else:
            if self.config.tensor_parallel_degree > 1:
                prefix_embedding = fleet.meta_parallel.VocabParallelEmbedding(
                    self.prefix_config.num_prefix_tokens,
                    self.prefix_config.hidden_size * self.prefix_config.num_hidden_layers * 2,
                )
            else:
                prefix_embedding = nn.Embedding(
                    self.prefix_config.num_prefix_tokens,
                    self.prefix_config.hidden_size * self.prefix_config.num_hidden_layers * 2,
                )
            prefix_encoder = nn.Sequential(prefix_embedding, prefix_dropout)
        return prefix_encoder

    def _get_past_key_values(self, batch_size):
        if self.inference:
            raise NotImplementedError("No support inference mode for PrefixModel")
        else:
            # (bs, prefixlen, hidden_dim*layer_num*2)
            past_key_values = self.prefix_encoder(self.prefix_tokens.unsqueeze(0).expand([batch_size, -1]))

            # (bs, prefixlen, hidden_dim*layer_num*2/tensor_parallel_degree)
            if self.config.tensor_parallel_degree > 1:
                split_past_key_values = past_key_values.split(axis=2)
                past_key_values = split_past_key_values[self.model.config.tensor_parallel_rank]

            # (bs, prefixlen, layer_num*2, head_num/tensor_parallel_degree,  head_dim)
            past_key_values = past_key_values.reshape(
                [
                    batch_size,
                    self.prefix_config.num_prefix_tokens,
                    self.prefix_config.num_hidden_layers * 2,
                    self.prefix_config.num_attention_heads // self.config.tensor_parallel_degree,
                    self.prefix_config.hidden_size // self.prefix_config.num_attention_heads,
                ]
            )

            if self.postprocess_past_key_value is not None:
                past_key_values = self.postprocess_past_key_value(past_key_values)

        return past_key_values

    def train(self):
        self.model.train()
        self.prefix_encoder.train()

    def eval(self):
        self.model.eval()
        self.prefix_encoder.eval()

    def get_model_trainable_state_dict(self):
        trainable_state_dict = OrderedDict()
        for name, weight in self.model.state_dict().items():
            if not weight.stop_gradient:
                trainable_state_dict[name] = weight
        return trainable_state_dict

    def print_trainable_parameters(self) -> None:
        freeze_numel = 0
        trainable_numel = 0
        for name, weight in self.model.state_dict().items():
            if weight.stop_gradient:
                freeze_numel += weight.numel().item()
            else:
                trainable_numel += weight.numel().item()
                print(name, weight.shape)
        for name, weight in self.prefix_encoder.state_dict().items():
            if weight.stop_gradient:
                freeze_numel += weight.numel().item()
            else:
                trainable_numel += weight.numel().item()
                print(name, weight.shape)
        logger.info(
            f"Frozen parameters: {freeze_numel:.2e} || Trainable parameters:{trainable_numel:.2e} || Total parameters:{freeze_numel+trainable_numel:.2e}|| Trainable:{trainable_numel / (freeze_numel+trainable_numel):.2%}"
        )

    @classmethod
    def from_pretrained(cls, model, prefix_path):
        prefix_config = PrefixConfig.from_pretrained(prefix_path)
        prefix_model = cls(model, prefix_config)
        # TODO(lugimzzz): support laod prefix_encoder parameter and past_key_values
        # TODO(lugimzzz): support mp
        return prefix_model

    def save_pretrained(self, save_directory: str, merge_tensor_parallel: bool = False):
        # TODO(lugimzzz): support load prefix_encoder parameter and past_key_values
        # TODO(lugimzzz): support mp
        assert not os.path.isfile(
            save_directory
        ), f"Saving directory ({save_directory}) should be a directory, not a file"
        os.makedirs(save_directory, exist_ok=True)
        if self.model.config.tensor_parallel_rank == 0:
            self.prefix_config.save_pretrained(save_directory)
            self.prefix_config.tensor_parallel_degree = self.model.config.tensor_parallel_degree


def bloom_postprocess_past_key_value(past_key_values):
    # (layer_num, bs, head_num/tensor_parallel_degree, prefixlen, head_dim)*2
    past_key_values = paddle.transpose(past_key_values, perm=[2, 0, 3, 1, 4]).split(2)
    # (layer_num, bs, head_num/tensor_parallel_degree, prefixlen, head_dim)
    num_hidden_layers, batch_size, num_attention_heads, num_prefix_tokens, head_hidden_size = past_key_values[0].shape
    # (layer_num, bs, prefixlen, head_num/tensor_parallel_degree, head_dim)
    keys, values = past_key_values[0].transpose([0, 1, 3, 2, 4]), past_key_values[1].transpose([0, 1, 3, 2, 4])
    # (layer_num, bs*head_num/tensor_parallel_degree, head_dim, prefixlen)
    keys = keys.reshape([num_hidden_layers, batch_size * num_attention_heads, head_hidden_size, num_prefix_tokens])
    # (layer_num, bs*head_num/tensor_parallel_degree, prefixlen, head_dim)
    values = values.reshape([num_hidden_layers, batch_size * num_attention_heads, num_prefix_tokens, head_hidden_size])

    return tuple(zip(keys, values))


def chatglm_postprocess_past_key_value(past_key_values):
    # (layer_num, prefixlen, bs, head_num/tensor_parallel_degree, head_dim)*2
    keys, values = paddle.transpose(past_key_values, perm=[2, 1, 0, 3, 4]).split(2)

    return tuple(zip(keys, values))


def llama_postprocess_past_key_value(past_key_values):
    # (layer_num, bs, prefixlen, head_num/tensor_parallel_degree, head_dim)*2
    keys, values = paddle.transpose(past_key_values, perm=[2, 0, 1, 3, 4]).split(2)

    return tuple(zip(keys, values))


def chatglm_pad_attention_mask(input_ids_shape, num_prefix_tokens, attention_mask):
    prefix_attention_mask = paddle.ones([input_ids_shape[0], 1, input_ids_shape[-1], num_prefix_tokens])
    prefix_attention_mask = (prefix_attention_mask < 0.5).astype("int64")
    return paddle.concat((prefix_attention_mask, attention_mask), axis=3)
