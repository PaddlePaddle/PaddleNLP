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
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Callable, Optional

import paddle
import paddle.nn as nn
from paddle.distributed import fleet

from ..transformers.model_utils import _add_variant, dtype_guard
from ..utils.distributed import distributed_gather
from ..utils.env import (
    PAST_KEY_VALUES_FILE_NAME,
    PREFIX_CONFIG_NAME,
    PREFIX_WEIGHT_FILE_NAME,
)
from ..utils.log import logger
from .prompt_utils import signature

__all__ = [
    "PrefixConfig",
    "PrefixModelForCausalLM",
]


@dataclass
class PrefixConfig:
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
    tensor_parallel_degree: int = field(default=-1, metadata={"help": ("1 for not use tensor parallel")})
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
        if self.prefix_config.dtype is None:
            self.prefix_config.dtype = paddle.get_default_dtype()
        with dtype_guard(self.prefix_config.dtype):
            self.prefix_encoder = self._create_prefix_encoder()
            self.prefix_dropout = nn.Dropout(p=prefix_config.prefix_dropout)
        self.prefix_tokens = paddle.arange(self.prefix_config.num_prefix_tokens, dtype="int64")
        self.model_prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
        self.inference = False
        self.postprocess_past_key_value = postprocess_past_key_value
        self.pad_attention_mask = pad_attention_mask
        if self.prefix_config.tensor_parallel_degree != self.model.config.tensor_parallel_degree:
            self.prefix_config.tensor_parallel_degree = self.model.config.tensor_parallel_degree
            logger.warning(
                f"Reset tensor_parallel_degree of prefix_config to {self.model.config.tensor_parallel_degree}."
            )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):

        batch_size = input_ids.shape[0]
        past_key_values = self._get_past_key_values(batch_size)

        if attention_mask is not None:
            if self.pad_attention_mask is not None:
                attention_mask = self.pad_attention_mask(
                    input_ids.shape, self.prefix_config.num_prefix_tokens, attention_mask
                )
            else:
                prefix_attention_mask = paddle.ones(
                    [batch_size, self.prefix_config.num_prefix_tokens], dtype=attention_mask.dtype
                )
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
                [model_kwargs["input_ids"].shape[0], self.prefix_config.num_prefix_tokens], dtype=attention_mask.dtype
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
        # freeze pretrained model
        for _, weight in self.model.state_dict().items():
            weight.stop_gradient = True
        # train prefix encoder only
        for _, weight in self.prefix_encoder.state_dict().items():
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

        # (bs, prefixlen, hidden_dim*layer_num*2)
        past_key_values = self.prefix_encoder(self.prefix_tokens.unsqueeze(0).expand([batch_size, -1]))

        # (bs, prefixlen, hidden_dim*layer_num*2/tensor_parallel_degree)
        if self.config.tensor_parallel_degree > 1:
            split_past_key_values = past_key_values.split(num_or_sections=self.config.tensor_parallel_degree, axis=2)
            past_key_values = split_past_key_values[self.model.config.tensor_parallel_rank]
            num_attention_heads = self.prefix_config.num_attention_heads // self.config.tensor_parallel_degree
        else:
            num_attention_heads = self.prefix_config.num_attention_heads

        # (bs, prefixlen, layer_num*2, head_num/tensor_parallel_degree,  head_dim)
        past_key_values = past_key_values.reshape(
            [
                batch_size,
                self.prefix_config.num_prefix_tokens,
                self.prefix_config.num_hidden_layers * 2,
                num_attention_heads,
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

    def print_trainable_parameters(self) -> None:
        freeze_numel = 0
        trainable_numel = 0
        for _, weight in self.model.state_dict().items():
            if weight.stop_gradient:
                freeze_numel += weight.numel().item()
            else:
                trainable_numel += weight.numel().item()
        for _, weight in self.prefix_encoder.state_dict().items():
            if weight.stop_gradient:
                freeze_numel += weight.numel().item()
            else:
                trainable_numel += weight.numel().item()
        logger.info(
            f"Frozen parameters: {freeze_numel:.2e} || Trainable parameters:{trainable_numel:.2e} || Total parameters:{freeze_numel+trainable_numel:.2e}|| Trainable:{trainable_numel / (freeze_numel+trainable_numel):.2%}"
        )

    @classmethod
    def from_pretrained(
        cls,
        model,
        prefix_path,
        postprocess_past_key_value=None,
        pad_attention_mask=None,
    ):
        # init prefix config & prefix model
        prefix_config = PrefixConfig.from_pretrained(prefix_path)
        # define a new variable to conserve original prefix_config.tensor_parallel_degree value which will update while initializing prefix model
        prefix_config_tensor_parallel_degree = prefix_config.tensor_parallel_degree
        prefix_model = cls(model, prefix_config, postprocess_past_key_value, pad_attention_mask)

        # define prefix weight name
        if prefix_config_tensor_parallel_degree > 1:
            prefix_weight_name = _add_variant(PREFIX_WEIGHT_FILE_NAME, f"tp{model.config.tensor_parallel_rank:0>2d}")
        else:
            prefix_weight_name = PREFIX_WEIGHT_FILE_NAME

        # load and set prefix weight parameter
        prefix_weight_path = os.path.join(prefix_path, prefix_weight_name)
        if os.path.exists(prefix_weight_path):
            # load prefix weight parameter
            prefix_state_dict = paddle.load(prefix_weight_path, return_numpy=True)
            logger.info(f"Loading the prefix weights from {prefix_weight_path}")

            if (
                prefix_config_tensor_parallel_degree > 1
                and prefix_config_tensor_parallel_degree != model.config.tensor_parallel_degree
            ):
                raise NotImplementedError(
                    f"{prefix_config_tensor_parallel_degree} is not equal to {model.config.tensor_parallel_degree}. Please merge prefix weights first."
                )

            # convert parameters to tensor parallel for mp model
            if prefix_config_tensor_parallel_degree <= 1 and model.config.tensor_parallel_degree > 1:
                prefix_state_dict = prefix_model._convert_tensor_parallel(prefix_state_dict=prefix_state_dict)

            # set prefix state dict
            prefix_model.set_state_dict(prefix_state_dict)
        else:
            logger.error(f"prefix weights not found under {prefix_path}, creating prefix weights from scratch")

        return prefix_model

    def save_pretrained(self, save_directory: str, merge_tensor_parallel: bool = False, **kwargs):
        variant = kwargs.get("variant", None)
        is_main_process = kwargs.get("is_main_process", paddle.distributed.get_rank() == 0)

        assert not os.path.isfile(
            save_directory
        ), f"Saving directory ({save_directory}) should be a directory, not a file"
        os.makedirs(save_directory, exist_ok=True)

        # past_key_values: (prefixlen, hidden_dim*layer_num*2)
        past_key_values = self.prefix_encoder(self.prefix_tokens.unsqueeze(0).expand([1, -1]))[0].numpy()

        if merge_tensor_parallel and self.model.config.tensor_parallel_degree > 1:
            trainable_state_dict = self.prefix_encoder.state_dict()
            trainable_state_dict = self._merge_trainable_tensor_parallel(trainable_state_dict)
            if not is_main_process:
                logger.info("Saving with merge_tensor_parallel, tensor_parallel_rank > 0 don't need save")
                return
            variant = None
            self.prefix_config.tensor_parallel_degree = -1
        else:
            trainable_state_dict = self.prefix_encoder.state_dict()
            if self.model.config.tensor_parallel_degree > 1:
                if variant is None:
                    variant = f"tp{self.model.config.tensor_parallel_rank:0>2d}"

        # save prefix tuning weight
        prefix_weight_name = _add_variant(PREFIX_WEIGHT_FILE_NAME, variant)
        weight_filename = os.path.join(save_directory, prefix_weight_name)
        paddle.save(trainable_state_dict, weight_filename)

        # save prefix config & past key values
        if is_main_process:
            self.prefix_config.save_pretrained(save_directory)
            self.prefix_config.tensor_parallel_degree = self.model.config.tensor_parallel_degree
            paddle.save({"past_key_values": past_key_values}, os.path.join(save_directory, PAST_KEY_VALUES_FILE_NAME))

    def set_state_dict(self, state_dict):
        self.prefix_encoder.set_state_dict(state_dict)
        logger.info("Load prefix weight successfully")

    def _merge_trainable_tensor_parallel(self, trainable_state_dict):
        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=False,
            tensor_parallel_degree=self.model.config.tensor_parallel_degree,
            tensor_parallel_rank=self.model.config.tensor_parallel_rank,
            num_attention_heads=self.model.config.num_attention_heads,
        )
        if self.prefix_config.prefix_projection:
            name_action_mappings = {
                "0.weight": partial(fn, is_column=False),
                "1.weight": partial(fn, is_column=True),
                "1.bias": partial(fn, is_column=True),
                "3.weight": partial(fn, is_column=False),
            }
        else:
            name_action_mappings = {
                "0.weight": partial(fn, is_column=False),
            }
        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        is_dst = paddle.distributed.get_rank(mp_group) == 0

        for key in trainable_state_dict:
            tensor = trainable_state_dict[key]
            if key in name_action_mappings:
                ret = distributed_gather(tensor, group=mp_group, offload=True)
                action = name_action_mappings[key]
                tensor = action(ret) if is_dst else None
                trainable_state_dict[key] = tensor
            else:
                trainable_state_dict[key] = tensor.numpy() if is_dst else None

        return trainable_state_dict

    def _convert_tensor_parallel(self, prefix_state_dict):
        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=True,
            tensor_parallel_degree=self.model.config.tensor_parallel_degree,
            tensor_parallel_rank=self.model.config.tensor_parallel_rank,
            num_attention_heads=self.model.config.num_attention_heads,
        )

        if self.prefix_config.prefix_projection:
            name_action_mappings = {
                "0.weight": partial(fn, is_column=False),
                "1.weight": partial(fn, is_column=True),
                "1.bias": partial(fn, is_column=True),
                "3.weight": partial(fn, is_column=False),
            }
        else:
            name_action_mappings = {
                "0.weight": partial(fn, is_column=False),
            }

        for name, action in name_action_mappings.items():
            tensor = prefix_state_dict.pop(name)
            prefix_state_dict[name] = action(tensor)
        return prefix_state_dict


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
    prefix_attention_mask = paddle.ones(
        [input_ids_shape[0], 1, input_ids_shape[-1], num_prefix_tokens], dtype=attention_mask.dtype
    )
    prefix_attention_mask = (prefix_attention_mask < 0.5).astype("int64")
    return paddle.concat((prefix_attention_mask, attention_mask), axis=3)
