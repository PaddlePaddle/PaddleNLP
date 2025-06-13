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
    GPTLMHeadAuto,
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
    def __init__(self, config, embedding_weights=None, ipp=None):
        super(GPTLMHeadAutoPP, self).__init__()
        self.config = config
        self.lm_head = GPTLMHeadAuto(config, embedding_weights=embedding_weights, ipp=ipp)

    def forward(self, args):
        hidden_states, attention_mask, position_ids = parse_args(args)
        logits = self.lm_head(hidden_states)
        return return_args(logits, attention_mask, position_ids)


class GPTDecoderLayerAutoPP(GPTPretrainedModelAuto):
    def __init__(self, config, layer_idx, ipp=None):
        super(GPTDecoderLayerAutoPP, self).__init__(config)
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
            # self.embeddings = GPTEmbeddingAutoPP(config)
            self.norm = GPTLayerNorm(config, config.hidden_size, epsilon=1e-5)
            self.lm_head = GPTLMHeadAutoPP(config, ipp=ipp)

    def forward(self, args):
        output_attentions = self.config.output_attentions
        use_cache = self.config.use_cache
        past_key_values = None
        output_hidden_states = self.config.output_hidden_states
        return_dict = self.config.return_dict
        if self.layer_idx == 0:
            input_ids, attention_mask, position_ids = parse_args(args)
            if self.config.sequence_parallel and use_cache:
                raise ValueError("We currently only support sequence parallel without cache.")
            if input_ids is not None:
                input_shape = input_ids.shape
                input_ids = input_ids.reshape((-1, input_shape[-1]))
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            # past_key_values is None
            if position_ids is None:
                past_length = 0
                position_ids = paddle.arange(past_length, input_shape[-1] + past_length, dtype="int64")
                position_ids = position_ids.unsqueeze(0)
                position_ids = paddle.expand(position_ids, input_shape)
            args = return_args(input_ids, attention_mask, position_ids)
            args = self.embeddings(args)
            hidden_states, attention_mask, position_ids = parse_args(args)

            length = input_shape[-1]
            cache_length = 0
            causal_mask = self.bias[:, :, cache_length:length, :length]
            if attention_mask is not None:
                if attention_mask.dtype != paddle.int64:
                    attention_mask = paddle.cast(attention_mask, dtype=paddle.int64)
                if len(attention_mask.shape) == 2:
                    attention_mask = attention_mask[:, None, None, :]
                attention_mask = (1.0 - (attention_mask & causal_mask)) * -1e4
            else:
                attention_mask = (1.0 - causal_mask) * -1e4
            # The tensor returned by triu not in static graph.
            attention_mask.stop_gradient = True
            args = return_args(hidden_states, attention_mask, position_ids)

        hidden_states, attention_mask, position_ids = parse_args(args)

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None
        has_gradient = not hidden_states.stop_gradient
        pre_ipp = None
        # if self.layer.ipp is not None and pre_ipp != self.ipp:
        #     hidden_states = dist.reshard(hidden_states, get_mesh(self.layer.ipp), [dist.Shard(0), dist.Replicate()])
        #     attention_mask = dist.reshard(
        #         attention_mask, get_mesh(self.layer.ipp), [dist.Replicate(), dist.Replicate()]
        #     )

        attention_mask = None
        if self.enable_recompute and has_gradient and self.config.recompute_granularity == "full":
            outputs = self.recompute_training(
                layer_module=self.layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=None,
                output_attentions=output_attentions,
            )
        else:
            outputs = self.layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
            )

        # outputs = hidden_states if both use_cache and output_attentions are False
        # Otherwise, outputs = (hidden_states, attention if output_attentions, cache if use_cache)
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
            output = self.norm(output)
            next_cache = next_decoder_cache if use_cache else None
            if not return_dict:
                temp_list = [output, next_cache, all_hidden_states, all_self_attentions]

                if not (use_cache or output_attentions or output_hidden_states):
                    outputs = output
                else:
                    outputs = tuple(v for v in temp_list if v is not None)

            if output_hidden_states:
                if return_dict:
                    outputs.hidden_states = (embedding_output,) + outputs.hidden_states
                else:  # outputs is a tuple
                    idx = 2 if use_cache else 1
                    all_hidden_states = (embedding_output,) + outputs[idx]
                    outputs[idx] = all_hidden_states

            if self.lm_head is not None:
                ret_args = self.lm_head(outputs)
            else:
                raise ValueError("ERR")

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
