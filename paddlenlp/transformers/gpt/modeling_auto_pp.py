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
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.pipelining.schedules import (
    Schedule1F1B,
    ScheduleFThenB,
    ScheduleVPP,
)
from paddle.distributed.auto_parallel.pipelining.stage import PipelineStage
from paddle.distributed.fleet.utils import recompute

from .configuration import GPTConfig
from .modeling_auto import (
    GPTDecoderLayerAuto,
    GPTEmbeddingsAuto,
    GPTLayerNorm,
    GPTLMHeadAuto,
    GPTPretrainedModelAuto,
)

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        mark_as_sequence_parallel_parameter,
    )
except:
    pass

__all__ = [
    "get_gpt_pp_schedule",
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


class GPTChunk(nn.Layer):
    def __init__(self, layers=None, is_first=False, is_last=False):
        super(GPTChunk, self).__init__()
        assert not (is_first and is_last)
        self.layers = layers
        self.is_first = is_first
        self.is_last = is_last

    def forward(self, *args, **kwargs):
        if self.is_first:
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            outputs = tuple([input_ids, attention_mask, position_ids])
            # decoder layers
            for idx, (decoder_layer) in enumerate(self.layers):
                outputs = decoder_layer(outputs)
            return outputs
        elif self.is_last:
            outputs = args
            # decoder layers
            for idx, (decoder_layer) in enumerate(self.layers):
                outputs = decoder_layer(outputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        else:
            outputs = args
            # decoder layers
            for idx, (decoder_layer) in enumerate(self.layers):
                outputs = decoder_layer(outputs)
        return outputs


def manual_model_split(model, stage_idx, group, mode, pp_degree):

    num_hidden_layers = model.config.num_hidden_layers
    virtual_pp_degree = model.config.virtual_pp_degree if mode == "VPP" else 1
    chunk_size = num_hidden_layers // virtual_pp_degree // pp_degree
    chunk_num = virtual_pp_degree * pp_degree
    layer_lists = None

    layer_lists = model.layers

    def _build_stage(model, stage_idx, group):
        new_model = None
        if stage_idx == 0:
            new_model = GPTChunk(layer_lists[:chunk_size], is_first=True, is_last=False)
        elif stage_idx == chunk_num - 1:
            new_model = GPTChunk(
                layer_lists[stage_idx * chunk_size : (stage_idx + 1) * chunk_size], is_first=False, is_last=True
            )
        else:
            new_model = GPTChunk(
                layer_lists[stage_idx * chunk_size : (stage_idx + 1) * chunk_size], is_first=False, is_last=False
            )
        stage = PipelineStage(new_model, stage_idx, chunk_num, group=group)
        return stage

    stages = []
    for i in range(virtual_pp_degree):
        stage = _build_stage(model, stage_idx + i * pp_degree, group)
        stages.append(stage)
    return stages


def get_gpt_pp_schedule(model, n_microbatches, loss_fn, mode, pp_degree, group):
    assert mode in ["VPP", "1F1B", "FThenB"]
    stages = manual_model_split(model, group.rank, group, mode, pp_degree)
    if mode == "VPP":
        schedule = ScheduleVPP(stages, n_microbatches=n_microbatches, loss_fn=loss_fn)
    elif mode == "1F1B":
        schedule = Schedule1F1B(stages[0], n_microbatches=n_microbatches, loss_fn=loss_fn)
    else:
        schedule = ScheduleFThenB(stages[0], n_microbatches=n_microbatches, loss_fn=loss_fn)
    return schedule


class GPTDecoderLayerAutoPP(nn.Layer):
    def __init__(self, config, layer_idx, ipp=None):
        super(GPTDecoderLayerAutoPP, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.embeddings = None
        self.norm = None
        self.lm_head = None
        if layer_idx == 0:
            self.embeddings = GPTEmbeddingsAuto(config)

        self.layer = GPTDecoderLayerAuto(config, ipp)
        self.ipp = ipp
        self.enable_recompute = False

        self.bias = paddle.tril(
            paddle.ones([1, 1, config.max_position_embeddings, config.max_position_embeddings], dtype="int64")
        )
        self.bias = dist.shard_tensor(self.bias, get_mesh(), [dist.Replicate(), dist.Replicate()])

        if layer_idx == config.num_hidden_layers - 1:
            self.norm = GPTLayerNorm(config, config.hidden_size, epsilon=1e-5)
            if config.sequence_parallel:
                mark_as_sequence_parallel_parameter(self.norm.weight)
                mark_as_sequence_parallel_parameter(self.norm.bias)
            self.lm_head = GPTLMHeadAuto(config, embedding_weights=None, ipp=ipp)

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        past_key_value: paddle.Tensor,
        attention_mask: paddle.Tensor,
        use_cache: bool,
        output_attentions: paddle.Tensor,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, output_attentions)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            use_cache,
            past_key_value,
            use_reentrant=self.config.recompute_use_reentrant,
        )
        return hidden_states

    def forward(self, args):
        output_attentions = self.config.output_attentions
        use_cache = self.config.use_cache
        if self.config.sequence_parallel and use_cache:
            raise ValueError("We currently only support sequence parallel without cache.")

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

            if position_ids is None:
                past_length = 0
                position_ids = paddle.arange(past_length, input_shape[-1] + past_length, dtype="int64")
                position_ids = position_ids.unsqueeze(0)
                position_ids = paddle.expand(position_ids, input_shape)
            args = return_args(input_ids, attention_mask, position_ids)
            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeddings=None)
            length = input_shape[-1]
            cache_length = 0
            causal_mask = self.bias[:, :, cache_length:length, :length]
            if not self.config.use_flash_attention:
                if attention_mask is not None:
                    if attention_mask.dtype != paddle.int64:
                        attention_mask = paddle.cast(attention_mask, dtype=paddle.int64)
                    if len(attention_mask.shape) == 2:
                        attention_mask = attention_mask[:, None, None, :]
                    attention_mask = (1.0 - (attention_mask & causal_mask)) * -1e4
                else:
                    attention_mask = (1.0 - causal_mask) * -1e4
            # The tensor returned by triu not in static graph.
            if attention_mask is not None:
                attention_mask.stop_gradient = True
            args = return_args(hidden_states, attention_mask, position_ids)

        hidden_states, attention_mask, position_ids = parse_args(args)
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None
        has_gradient = not hidden_states.stop_gradient
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

            if self.lm_head is not None:
                logits = self.lm_head(outputs)
                ret_args = return_args(
                    logits,
                )

        return ret_args


class GPTForCausalLMAutoPP(GPTPretrainedModelAuto):
    def __init__(self, config: GPTConfig):
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

        # decoder layers
        for layer in self.layers:
            outputs = layer(outputs)

        return outputs[0]
