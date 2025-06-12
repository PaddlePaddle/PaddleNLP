# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed import fleet

from ..model_outputs import BaseModelOutputWithPastAndCrossAttentions
from .modeling_auto import (
    GPTDecoderLayerAuto,
    GPTEmbeddingsAuto,
    GPTLayerNorm,
    GPTPretrainedModelAuto,
    MultiHeadAttentionAuto,
    seed_guard_context,
)

__all__ = [
    "GPTForCausalLMAutoPP",
]


def parse_args(args):
    hidden_states, attention_mask, position_ids = None, None, None
    if isinstance(args, tuple):
        if len(args) == 3:
            hidden_states, attention_mask, position_ids = args
        elif len(args) == 2:
            hidden_states, attention_mask = args
        elif len(args) == 1:
            hidden_states = args[0]
    else:
        hidden_states = args
    if position_ids is not None:
        position_ids.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    return hidden_states, attention_mask, position_ids


def return_args(hidden_states, attention_mask=None, position_ids=None):
    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if position_ids is not None:
        ret += (position_ids.clone(),)
    if len(ret) == 1:
        ret = ret[0]

    return ret


def global_mesh_starts_with_pp():
    mesh = fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        return mesh.get_mesh_with_dim("pp")
    else:
        return mesh


def get_mesh(pp_idx=0):
    mesh = fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp")[pp_idx]
    return mesh


class GPTEmbeddingAutoPP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = GPTEmbeddingsAuto(config)

    def forward(self, args):
        input_ids, position_ids, _ = parse_args(args)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeddings=None)
        return return_args(embedding_output, None, position_ids)


class GPTLMHeadAutoPP(nn.Layer):
    def __init__(self, config, ipp=None):
        super(GPTLMHeadAutoPP, self).__init__()
        self.config = config
        self.transpose_y = True
        self.ipp = ipp
        self.weight = self.create_parameter(
            shape=[config.vocab_size, config.hidden_size], dtype=paddle.get_default_dtype()
        )
        self.weight = dist.shard_tensor(self.weight, get_mesh(ipp), [dist.Replicate(), dist.Shard(0)])

    def forward(self, args):
        hidden_states, attention_mask, position_ids = parse_args(args)

        # if self.config.sequence_parallel:
        #     hidden_states = dist.reshard(hidden_states, get_mesh(self.ipp), [dist.Replicate(), dist.Replicate()])
        #     hidden_states = paddle.reshape(hidden_states, [-1, self.config.seq_length, self.config.hidden_size])

        y = dist.reshard(self.weight, get_mesh(self.ipp), [dist.Replicate(), dist.Shard(0)])
        logits = paddle.matmul(hidden_states, y, transpose_y=self.transpose_y)

        return return_args(logits, attention_mask, position_ids)


class GPTDecoderLayerAutoPP(nn.Layer):
    def __init__(self, config, layer_idx, ipp=None):
        super(GPTDecoderLayerAutoPP, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.embeddings = None
        self.lm_head = None
        self.norm = None
        if layer_idx == 0:
            self.embeddings = GPTEmbeddingAutoPP(config)

        self.layer = GPTDecoderLayerAuto(config, ipp)
        self.ipp = ipp
        self.enable_recompute = False
        self.bias = paddle.tril(
            paddle.ones([1, 1, config.max_position_embeddings, config.max_position_embeddings], dtype="int64")
        )
        self.bias = dist.shard_tensor(self.bias, get_mesh(), [dist.Replicate(), dist.Replicate()])

        if layer_idx == config.num_hidden_layers - 1:
            self.norm = GPTLayerNorm(config, config.hidden_size, epsilon=1e-5)
            self.lm_head = GPTLMHeadAutoPP(config, ipp)

    def forward(self, args):
        if self.embeddings is not None:
            input_ids, attention_mask, position_ids = parse_args(args)
            input_shape = input_ids.shape
            input_ids = input_ids.reshape((-1, input_shape[-1]))
            if position_ids is None:
                past_length = 0
                # if past_key_values[0] is not None:
                #     # bs, seq_len, num_head, head_dim
                #     past_length = past_key_values[0][0].shape[1]
                position_ids = paddle.arange(past_length, input_shape[-1] + past_length, dtype="int64")
                position_ids = position_ids.unsqueeze(0)
                position_ids = paddle.expand(position_ids, input_shape)
            args = return_args(input_ids, attention_mask, position_ids)
            args = self.embeddings(args)
            length = input_shape[-1]

        hidden_states, attention_mask, position_ids = parse_args(args)

        output_attentions = self.config.output_attentions
        use_cache = self.config.use_cache

        past_key_values = None
        inputs_embeds = None

        # cache_length = 0
        # causal_mask = self.bias[:, :, cache_length:length, :length]
        # attention_mask = (1.0 - causal_mask) * -1e4
        # attention_mask.stop_gradient = True
        attention_mask = None
        output_hidden_states = False
        return_dict = False
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None
        outputs = self.layer(
            hidden_states,
            attention_mask,
            use_cache,
            past_key_values,
            output_attentions,
        )
        output = outputs[0] if (use_cache or output_attentions) else outputs
        all_self_attentions = all_self_attentions + (outputs[1],) if output_attentions else None
        all_hidden_states = all_hidden_states + (output,) if output_hidden_states else None
        next_decoder_cache = next_decoder_cache + (outputs[-1],) if use_cache else None
        ret_args = return_args(
            output,
            attention_mask,
            position_ids,
        )
        if self.norm is not None:
            ret_args = self.norm(output)
        if self.lm_head is not None:
            ret_args = self.lm_head(ret_args)
        return ret_args


class GPTForCausalLMAutoPP(GPTPretrainedModelAuto):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []

        decoder_layers = []

        def get_pp_stage_id(layer_id):
            pp_degree = global_mesh_starts_with_pp().shape[0]
            chunk_size = self.config.num_hidden_layers // (pp_degree * self.config.virtual_pp_degree)
            chunk_id = layer_id // chunk_size
            pp_stage_id = chunk_id % pp_degree
            return pp_stage_id

        for i in range(config.num_hidden_layers):
            pp_stage_id = get_pp_stage_id(i)
            decoder_layers.append(GPTDecoderLayerAutoPP(config, i, pp_stage_id))
        self.layers = nn.LayerList(decoder_layers)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        outputs = return_args(input_ids, attention_mask, position_ids)

        for layer in self.layers:
            outputs = layer(outputs)

        return outputs[0]
