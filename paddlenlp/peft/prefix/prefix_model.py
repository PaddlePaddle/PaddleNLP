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

import os
from functools import partial
from typing import Callable, Optional

import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed import fleet

from ...prompt.prompt_utils import signature
from ...transformers.model_utils import _add_variant, dtype_guard
from ...utils.distributed import distributed_gather
from ...utils.env import PAST_KEY_VALUES_FILE_NAME, PREFIX_WEIGHTS_NAME
from ...utils.log import logger
from .prefix_config import PrefixConfig


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
        if self.model.base_model_prefix == "chatglm_v2":
            self.prefix_config.tensor_parallel_degree = -1
        else:
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
                if len(attention_mask.shape) == 2:
                    prefix_attention_mask = paddle.ones(
                        [batch_size, self.prefix_config.num_prefix_tokens], dtype=attention_mask.dtype
                    )
                elif len(attention_mask.shape) == 3:
                    batch_size, src_seq_len, tgt_seq_len = attention_mask.shape
                    prefix_attention_mask = paddle.ones(
                        [batch_size, src_seq_len, self.prefix_config.num_prefix_tokens], dtype=attention_mask.dtype
                    )
                elif len(attention_mask.shape) == 4:
                    batch_size, num_heads, src_seq_len, tgt_seq_len = attention_mask.shape
                    prefix_attention_mask = paddle.ones(
                        [batch_size, num_heads, src_seq_len, self.prefix_config.num_prefix_tokens],
                        dtype=attention_mask.dtype,
                    )
                else:
                    raise ValueError(f"Unexpected attention_mask shape: {attention_mask.shape}")
                attention_mask = paddle.concat((prefix_attention_mask, attention_mask), axis=-1)
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
        batch_size = model_kwargs["input_ids"].shape[0]
        if self.pad_attention_mask is not None:
            attention_mask = self.pad_attention_mask(
                model_kwargs["input_ids"].shape, self.prefix_config.num_prefix_tokens, attention_mask
            )
        else:
            if len(attention_mask.shape) == 2:
                prefix_attention_mask = paddle.ones(
                    [batch_size, self.prefix_config.num_prefix_tokens], dtype=attention_mask.dtype
                )
            elif len(attention_mask.shape) == 3:
                batch_size, src_seq_len, tgt_seq_len = attention_mask.shape
                prefix_attention_mask = paddle.ones(
                    [batch_size, src_seq_len, self.prefix_config.num_prefix_tokens], dtype=attention_mask.dtype
                )
            elif len(attention_mask.shape) == 4:
                batch_size, num_heads, src_seq_len, tgt_seq_len = attention_mask.shape
                prefix_attention_mask = paddle.ones(
                    [batch_size, num_heads, src_seq_len, self.prefix_config.num_prefix_tokens],
                    dtype=attention_mask.dtype,
                )
            else:
                raise ValueError(f"Unexpected attention_mask shape: {attention_mask.shape}")
            attention_mask = paddle.concat((prefix_attention_mask, attention_mask), axis=-1)
        model_kwargs["attention_mask"] = attention_mask

        if "past_key_values" in self.forward_keys:
            key = "past_key_values"
        elif "cache" in self.forward_keys:
            key = "cache"
        else:
            raise NotImplementedError("Model does not support past_key_values either cache")
        if model_kwargs[key] is None:
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
        self.head_dim = self.prefix_config.hidden_size // self.prefix_config.num_attention_heads
        if self.prefix_config.multi_query_group_num is not None:
            self.num_heads = self.prefix_config.multi_query_group_num
        else:
            self.num_heads = self.prefix_config.num_attention_heads
        if self.prefix_config.prefix_projection:
            activation = nn.Tanh()
            if self.prefix_config.tensor_parallel_degree > 1:
                prefix_embedding = fleet.meta_parallel.VocabParallelEmbedding(
                    self.prefix_config.num_prefix_tokens,
                    self.head_dim * self.num_heads,
                )
                prefix_proj_0 = fleet.meta_parallel.ColumnParallelLinear(
                    self.head_dim * self.num_heads,
                    self.prefix_config.prefix_projection_hidden_size,
                    has_bias=True,
                    gather_output=False,
                )
                prefix_proj_1 = fleet.meta_parallel.RowParallelLinear(
                    self.prefix_config.prefix_projection_hidden_size,
                    self.head_dim * self.num_heads * self.prefix_config.num_hidden_layers * 2,
                    has_bias=True,
                    input_is_parallel=True,
                )
            else:
                prefix_embedding = nn.Embedding(
                    self.prefix_config.num_prefix_tokens,
                    self.head_dim * self.num_heads,
                )
                prefix_proj_0 = nn.Linear(
                    self.head_dim * self.num_heads,
                    self.prefix_config.prefix_projection_hidden_size,
                )
                prefix_proj_1 = nn.Linear(
                    self.prefix_config.prefix_projection_hidden_size,
                    self.head_dim * self.num_heads * self.prefix_config.num_hidden_layers * 2,
                )
            prefix_encoder = nn.Sequential(prefix_embedding, prefix_proj_0, activation, prefix_proj_1, prefix_dropout)
        else:
            if self.prefix_config.tensor_parallel_degree > 1:
                prefix_embedding = fleet.meta_parallel.VocabParallelEmbedding(
                    self.prefix_config.num_prefix_tokens,
                    self.head_dim * self.num_heads * self.prefix_config.num_hidden_layers * 2,
                )
            else:
                prefix_embedding = nn.Embedding(
                    self.prefix_config.num_prefix_tokens,
                    self.head_dim * self.num_heads * self.prefix_config.num_hidden_layers * 2,
                )
            prefix_encoder = nn.Sequential(prefix_embedding, prefix_dropout)
        return prefix_encoder

    def _get_past_key_values(self, batch_size):

        # (bs, prefixlen, hidden_dim*layer_num*2)
        past_key_values = self.prefix_encoder(self.prefix_tokens.unsqueeze(0).expand([batch_size, -1]))

        # (bs, prefixlen, hidden_dim*layer_num*2/tensor_parallel_degree)
        if self.prefix_config.tensor_parallel_degree > 1:
            split_past_key_values = past_key_values.split(
                num_or_sections=self.prefix_config.tensor_parallel_degree, axis=2
            )
            past_key_values = split_past_key_values[self.model.config.tensor_parallel_rank]
            num_heads_per_partition = self.num_heads // self.prefix_config.tensor_parallel_degree
        else:
            num_heads_per_partition = self.num_heads

        # (bs, prefixlen, layer_num*2, head_num/tensor_parallel_degree,  head_dim)
        past_key_values = past_key_values.reshape(
            [
                batch_size,
                self.prefix_config.num_prefix_tokens,
                self.prefix_config.num_hidden_layers * 2,
                num_heads_per_partition,
                self.head_dim,
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
            prefix_weight_name = _add_variant(PREFIX_WEIGHTS_NAME, f"tp{model.config.tensor_parallel_rank:0>2d}")
        else:
            prefix_weight_name = PREFIX_WEIGHTS_NAME

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

    def save_pretrained(self, save_directory: str, merge_tensor_parallel: bool = True, **kwargs):
        variant = kwargs.get("variant", None)
        is_main_process = kwargs.get("is_main_process", paddle.distributed.get_rank() == 0)

        assert not os.path.isfile(
            save_directory
        ), f"Saving directory ({save_directory}) should be a directory, not a file"
        os.makedirs(save_directory, exist_ok=True)

        # past_key_values: (prefixlen, hidden_dim*layer_num*2)
        past_key_values = self.prefix_encoder(self.prefix_tokens.unsqueeze(0).expand([1, -1]))
        # (prefixlen, 2, layer_num, num_heads, head_dim)
        past_key_values = past_key_values.reshape(
            [
                self.prefix_config.num_prefix_tokens,
                2,
                self.prefix_config.num_hidden_layers,
                self.num_heads,
                self.head_dim,
            ]
        )
        # (num_layers, 2, num_heads, prefixlen, head_dim)
        past_key_values = paddle.transpose(past_key_values, perm=[2, 1, 3, 0, 4]).numpy()

        if merge_tensor_parallel and self.prefix_config.tensor_parallel_degree > 1:
            trainable_state_dict = self.prefix_encoder.state_dict()
            trainable_state_dict = self._merge_trainable_tensor_parallel(trainable_state_dict)
            if not is_main_process:
                logger.info("Saving with merge_tensor_parallel, tensor_parallel_rank > 0 don't need save")
                return
            variant = None
            self.prefix_config.tensor_parallel_degree = -1
        else:
            trainable_state_dict = self.prefix_encoder.state_dict()
            if self.prefix_config.tensor_parallel_degree > 1:
                if variant is None:
                    variant = f"tp{self.model.config.tensor_parallel_rank:0>2d}"

        # save prefix tuning weight
        prefix_weight_name = _add_variant(PREFIX_WEIGHTS_NAME, variant)
        weight_filename = os.path.join(save_directory, prefix_weight_name)
        paddle.save(trainable_state_dict, weight_filename)

        # save prefix config & past key values
        if is_main_process:
            self.prefix_config.save_pretrained(save_directory)
            np.save(os.path.join(save_directory, PAST_KEY_VALUES_FILE_NAME), past_key_values)

        if self.model.base_model_prefix == "chatglm_v2":
            self.prefix_config.tensor_parallel_degree = -1
        else:
            self.prefix_config.tensor_parallel_degree = self.model.config.tensor_parallel_degree

    def set_state_dict(self, state_dict):
        self.prefix_encoder.set_state_dict(state_dict)
        logger.info("Load prefix weight successfully")

    def _merge_trainable_tensor_parallel(self, trainable_state_dict):
        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=False,
            tensor_parallel_degree=self.prefix_config.tensor_parallel_degree,
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
