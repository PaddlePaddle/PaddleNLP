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
    GPTConfig,
    GPTDecoderLayer,
    GPTEmbeddings,
    GPTPretrainedModel,
    GPTPretrainingCriterion,
    parallel_matmul,
)

__all__ = [
    "GPTForCausalLMPipe",
]


def get_hcg():
    return fleet.get_hybrid_communicate_group()


def get_attr(layer, name):
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


def parse_args(args):
    if isinstance(args, tuple):
        if len(args) == 3:
            hidden_states, attention_mask, position_ids = args
        elif len(args) == 2:
            hidden_states, attention_mask = args
            position_ids = None
    else:
        hidden_states = args
        attention_mask, position_ids = None, None

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


class GPTEmbeddingPipe(GPTEmbeddings):
    """Extends GPTEmbeddings to forward attention_mask through the pipeline."""

    @property
    def embedding_weight(self):
        return get_attr(self.word_embeddings, "weight")

    def forward(self, args):
        input_ids, attention_mask, position_ids = parse_args(args)
        input_ids.stop_gradient = True
        embeddings = super().forward(input_ids=input_ids, position_ids=position_ids)
        return embeddings


class GPTDecoderLayerPipe(GPTDecoderLayer):
    def forward(self, args):
        hidden_states, attention_mask, position_ids = parse_args(args)
        # hidden_states = super().forward(hidden_states, tgt_mask=attention_mask)
        if self.enable_recompute and self.config.recompute_granularity == "full":
            hidden_states = recompute(super().forward, hidden_states, attention_mask)
        else:
            hidden_states = super().forward(hidden_states, tgt_mask=attention_mask)

        return return_args(hidden_states, attention_mask, position_ids)


class LayerNormPipe(nn.LayerNorm):
    def __init__(self, config):
        super(LayerNormPipe, self).__init__(config.hidden_size, epsilon=1e-05)

    def forward(self, args):
        hidden_states, attention_mask, position_ids = parse_args(args)
        hidden_states = super().forward(hidden_states)
        return return_args(hidden_states, attention_mask, position_ids)


class GPTForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """LlamaForPretraining adapted for pipeline parallelism.

    The largest change is flattening the LlamaModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    config_class = GPTConfig

    _get_tensor_parallel_mappings = GPTPretrainedModel._get_tensor_parallel_mappings
    _init_weights = GPTPretrainedModel._init_weights

    pretrained_init_configuration = GPTPretrainedModel.pretrained_init_configuration
    pretrained_resource_files_map = GPTPretrainedModel.pretrained_resource_files_map

    # NO base_model_prefix !!!!

    def __init__(
        self,
        config,
        pp_recompute_interval=1,
    ):
        self.config = config

        virtual_pp_degree = getattr(self.config, "virtual_pp_degree", 1)

        hcg = get_hcg()
        tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)

        config.tensor_parallel_degree = tensor_parallel_degree
        config.tensor_parallel_rank = tensor_parallel_rank

        self.add_sequential_layer(
            SharedLayerDesc("gpt", GPTEmbeddingPipe, shared_weight_attr="embedding_weight", config=config),
            "gpt.embeddings",
        )
        for i in range(config.num_hidden_layers):
            self.add_sequential_layer(
                LayerDesc(GPTDecoderLayerPipe, config=config),
                f"gpt.decoder.layers.{i}",
            )

        self.add_sequential_layer(LayerDesc(LayerNormPipe, config=config), "gpt.decoder.norm")

        def _logits_helper(embedding, output):
            return parallel_matmul(output, embedding.embedding_weight, True)

        self.add_sequential_layer(
            SharedLayerDesc(
                "gpt",
                GPTEmbeddingPipe,
                forward_func=_logits_helper,
                shared_weight_attr="embedding_weight",
                config=config,
            ),
            "gpt.embeddings",
        )

        recompute_interval = 0
        # if use_recompute and recompute_granularity == "full":
        #    assert pp_recompute_interval <= config.num_hidden_layers // (
        #        virtual_pp_degree * get_hcg().topology().get_dim_size("pipe")
        #    ), "pp recompute interval should smaller than num layers of each pp chunk"
        #    recompute_interval = pp_recompute_interval

        seg_method = "layer:GPTDecoderLayer"
        if config.num_hidden_layers % get_hcg().topology().get_dim_size("pipe") != 0:
            seg_method = "uniform"

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=GPTPretrainingCriterion(config),
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
