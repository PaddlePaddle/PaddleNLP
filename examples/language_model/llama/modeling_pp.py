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

# pass
import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer

from paddlenlp.transformers import PretrainedModel
from paddlenlp.transformers.llama.modeling import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaLMHead,
    LlamaPretrainingCriterion,
    RMSNorm,
)


def get_hcg():
    return fleet.get_hybrid_communicate_group()


class LlamaEmbedding(nn.Layer):
    """Extends LlamaEmbeddings to forward attention_mask through the pipeline."""

    def __init__(self, config):
        super(LlamaEmbedding, self).__init__()
        if config.tensor_parallel_degree > 1:
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.embed_tokens(input_ids)


class PipelinePretrainedModel(PretrainedModel):
    _sequential_layers = []
    _pipeline_name_mapping = None

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def add_sequential_layer(self, layer_desc, name_prefix=""):
        self._sequential_layers.append({"layer": layer_desc, "name_prefix": name_prefix})

    def get_sequential_layers(self):
        return [x["layer"] for x in self._sequential_layers]

    def get_sequential_name_prefixs(self):
        return {str(index): x["name_prefix"] for index, x in enumerate(self._sequential_layers)}

    def _set_pipeline_name_mapping(self, mappings=None):
        if mappings is not None:
            self._pipeline_name_mapping = mappings
        else:
            mapping = {}
            state_dict_keys = list(super().state_dict().keys())
            prefixs = self.get_sequential_name_prefixs()
            for k in state_dict_keys:
                name_splited = k.split(".")
                name_splited[0] = prefixs[name_splited[0]]
                mapping[".".join(name_splited)] = k
            self._pipeline_name_mapping = mapping

        return self._pipeline_name_mapping

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        prefixs = self.get_sequential_name_prefixs()
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            name_splited = k.split(".")
            name_splited[0] = prefixs[name_splited[0]]
            state_dict[".".join(name_splited)] = v

        return state_dict

    def set_state_dict(self, state_dict, *args, **kwargs):
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        assert len(self._pipeline_name_mapping) > 0, "The pipeline stage must have parameters!"

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if k not in self._pipeline_name_mapping:
                continue
            state_dict[self._pipeline_name_mapping[k]] = v

        return super().set_state_dict(state_dict, *args, **kwargs)


class LlamaForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """LlamaForPretraining adapted for pipeline parallelism.

    The largest change is flattening the LlamaModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    config_class = LlamaConfig

    # NO base_model_prefix !!!!

    def __init__(
        self,
        config,
        # num_partitions=1,
        # topology=None,
        use_recompute=False,
        # fused_linear=False,
        # fuse_attn_qkv=False,
        # scale_qk_by_layer_num=True,
        recompute_granularity="full",
        virtual_pp_degree=1,
        # sequence_parallel=False,
        # no_recompute_layers=None,
        pp_recompute_interval=1,
        # use_flash_attn=False,
        # fused_softmax_with_triangular=False,
    ):
        self.config = config

        hcg = get_hcg()
        tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)

        config.tensor_parallel_degree = tensor_parallel_degree
        config.tensor_parallel_rank = tensor_parallel_rank

        self.add_sequential_layer(LayerDesc(LlamaEmbedding, config=config), "llama")
        for i in range(config.num_hidden_layers):
            self.add_sequential_layer(LayerDesc(LlamaDecoderLayer, config=config), f"llama.layers.{i}")

        self.add_sequential_layer(LayerDesc(RMSNorm, config=config), "llama.norm")
        self.add_sequential_layer(LayerDesc(LlamaLMHead, config=config), "lm_head")

        recompute_interval = 0
        if use_recompute and recompute_granularity == "full":
            assert pp_recompute_interval <= config.num_hidden_layers // (
                virtual_pp_degree * get_hcg().topology().get_dim_size("pipe")
            ), "pp recompute interval should smaller than num layers of each pp chunk"
            recompute_interval = pp_recompute_interval

        seg_method = "layer:LlamaDecoderLayer"
        if config.num_hidden_layers % get_hcg().topology().get_dim_size("pipe") != 0:
            seg_method = "uniform"

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=LlamaPretrainingCriterion(config),
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
        # DON'T init PipelinePretrainedModel
        # PipelinePretrainedModel.__init__(self.super(), config=config)
