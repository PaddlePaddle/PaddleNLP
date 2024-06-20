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

import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.distributed.fleet.utils import recompute

from paddlenlp.transformers.model_utils import PipelinePretrainedModel

from .modeling import (
    GemmaConfig,
    GemmaDecoderLayer,
    GemmaLMHead,
    GemmaModel,
    GemmaPretrainedModel,
    GemmaPretrainingCriterion,
    GemmaRMSNorm,
    build_alibi_tensor,
)


def __repr__(self):
    return self.layer_func.__name__


# hack LayerDesc for showing to much config
LayerDesc.__repr__ = __repr__

__all__ = [
    "GemmaForCausalLMPipe",
]


def parse_args(args):
    if isinstance(args, tuple):
        if len(args) == 4:
            hidden_states, attention_mask, position_ids, alibi = args
        if len(args) == 3:
            hidden_states, attention_mask, position_ids = args
            alibi = None
        elif len(args) == 2:
            hidden_states, attention_mask = args
            position_ids = None
            alibi = None
    else:
        hidden_states = args
        attention_mask, position_ids, alibi = None, None, None

    if position_ids is not None:
        position_ids.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    if alibi is not None:
        alibi.stop_gradient = True

    return hidden_states, attention_mask, position_ids, alibi


def return_args(hidden_states, attention_mask=None, position_ids=None, alibi=None):
    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if position_ids is not None:
        ret += (position_ids.clone(),)
    if alibi is not None:
        ret += (alibi.clone(),)

    if len(ret) == 1:
        ret = ret[0]

    return ret


def get_attr(layer, name):
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


class GemmaEmbeddingPipe(nn.Layer):
    """Extends GemmaEmbeddings to forward attention_mask through the pipeline."""

    def __init__(self, config):
        super(GemmaEmbeddingPipe, self).__init__()
        self.config = config
        self.sequence_parallel = config.sequence_parallel
        self.hidden_size = config.hidden_size
        if config.tensor_parallel_degree > 1:
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    @property
    def embedding_weight(self):
        return get_attr(self.embed_tokens, "weight")

    def forward(self, args):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_ids, attention_mask, position_ids, alibi = parse_args(args)
        input_embeds = self.embed_tokens(input_ids)
        if self.sequence_parallel:
            from paddlenlp.transformers import ScatterOp

            # [bs, seq_len, num_head * head_dim] -> [bs * seq_len, num_head * head_dim]
            bs, seq_len, hidden_size = input_embeds.shape
            input_embeds = paddle.reshape_(input_embeds, [bs * seq_len, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            input_embeds = ScatterOp.apply(input_embeds)

        batch_size, seq_length = input_ids.shape
        alibi = None
        if self.config.alibi:
            # embed positions
            mask = (
                attention_mask
                if attention_mask is not None
                else paddle.ones((batch_size, seq_length), dtype=paddle.bool)
            )
            alibi = build_alibi_tensor(mask, self.config.num_attention_heads, dtype=input_embeds.dtype)

            if self.config.tensor_parallel_degree > 1:
                block_size = self.config.num_attention_heads // self.config.tensor_parallel_degree
                alibi = alibi[
                    :,
                    self.config.tensor_parallel_rank
                    * block_size : (self.config.tensor_parallel_rank + 1)
                    * block_size,
                ]
                alibi = alibi.reshape([batch_size * block_size, 1, seq_length])
            else:
                alibi = alibi.reshape([batch_size * self.config.num_attention_heads, 1, seq_length])
            alibi.stop_gradient = True

        if attention_mask is not None:
            attention_mask = GemmaModel._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), 0, input_embeds.dtype
            )
            attention_mask.stop_gradient = True

        if self.config.alibi and attention_mask is None:
            attention_mask = GemmaModel._prepare_decoder_attention_mask(
                None, (batch_size, seq_length), 0, input_embeds.dtype
            )
            attention_mask.stop_gradient = True

        hidden_states = input_embeds * (self.config.hidden_size**0.5)
        return return_args(hidden_states, attention_mask, position_ids, alibi)


class GemmaDecoderLayerPipe(GemmaDecoderLayer):
    def forward(self, args):
        hidden_states, attention_mask, position_ids, alibi = parse_args(args)
        # we can't distinguish
        # hidden_states, attention_mask, position_ids or
        # hidden_states, attention_mask, alibi
        if self.config.alibi and alibi is None and position_ids is not None:
            alibi = position_ids
            position_ids = None

        has_gradient = not hidden_states.stop_gradient
        if self.enable_recompute and self.config.recompute_granularity == "full" and has_gradient:
            if attention_mask is not None or alibi is not None:
                hidden_states = recompute(
                    super().forward, hidden_states, attention_mask=attention_mask, alibi=alibi, use_reentrant=False
                )
            else:
                # for pretrain
                hidden_states = recompute(
                    super().forward, hidden_states, use_reentrant=self.config.recompute_use_reentrant
                )
        else:
            hidden_states = super().forward(hidden_states, attention_mask=attention_mask, alibi=alibi)

        return return_args(hidden_states, attention_mask, position_ids, alibi)


class GemmaRMSNormPipe(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.norm = GemmaRMSNorm(config)

    def forward(self, args):
        hidden_states, attention_mask, position_ids, alibi = parse_args(args)
        return self.norm(hidden_states)


class GemmaLMHeadPipe(GemmaLMHead):
    def __init__(self, config):
        super(GemmaLMHeadPipe, self).__init__(config)

    @property
    def embedding_weight(self):
        return get_attr(self, "weight")


class GemmaForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """GemmaForPretraining adapted for pipeline parallelism.

    The largest change is flattening the GemmaModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    config_class = GemmaConfig

    _get_tensor_parallel_mappings = GemmaPretrainedModel._get_tensor_parallel_mappings
    _init_weights = GemmaPretrainedModel._init_weights
    _keys_to_ignore_on_load_unexpected = GemmaPretrainedModel._keys_to_ignore_on_load_unexpected

    # DONOT Add base_model_prefix !!!!

    def __init__(self, config):
        self.config = config

        self.recompute_granularity = self.config.recompute_granularity
        self.pp_recompute_interval = self.config.pp_recompute_interval
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []
        if self.recompute_granularity == "full":
            assert len(self.no_recompute_layers) == 0, "for pp with full recompute, no_recompute_layers is not support"

        virtual_pp_degree = getattr(self.config, "virtual_pp_degree", 1)

        def get_hcg():
            return fleet.get_hybrid_communicate_group()

        hcg = get_hcg()
        tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)

        # TODO: fix tensor_parallel_degree rewrite in here
        config.tensor_parallel_degree = tensor_parallel_degree
        config.tensor_parallel_rank = tensor_parallel_rank

        self.add_sequential_layer(
            SharedLayerDesc(
                key="gemma_weigt_share",
                layer_func=GemmaEmbeddingPipe,
                shared_weight_attr="embedding_weight",
                config=config,
            ),
            "gemma",
        )
        for i in range(config.num_hidden_layers):
            self.add_sequential_layer(
                LayerDesc(GemmaDecoderLayerPipe, config=config, layerwise_recompute=i not in self.no_recompute_layers),
                f"gemma.layers.{i}",
            )

        self.add_sequential_layer(LayerDesc(GemmaRMSNormPipe, config=config), "gemma")
        self.add_sequential_layer(
            SharedLayerDesc(
                key="gemma_weigt_share",
                layer_func=GemmaLMHeadPipe,
                shared_weight_attr="embedding_weight",
                config=config,
            ),
            "lm_head",
        )

        recompute_interval = 0

        seg_method = "layer:GemmaDecoderLayer"
        if config.num_hidden_layers % get_hcg().topology().get_dim_size("pipe") != 0:
            seg_method = "uniform"

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=GemmaPretrainingCriterion(config),
            topology=get_hcg().topology(),
            seg_method=seg_method,
            recompute_interval=recompute_interval,
            recompute_ctx={
                "mp_group": get_hcg().get_model_parallel_group(),
                "offload": False,
                "partition": False,
            },
            num_virtual_pipeline_stages=virtual_pp_degree,
        )
        self.apply(self._init_weights)
        # DON'T init PipelinePretrainedModel
        # PipelinePretrainedModel.__init__(self.super(), config=config)
