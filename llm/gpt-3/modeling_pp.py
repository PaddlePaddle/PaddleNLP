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
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.distributed.fleet.utils import recompute

from paddlenlp.transformers import (
    GPTConfig,
    GPTDecoderLayer,
    GPTEmbeddings,
    GPTPretrainedModel,
    GPTPretrainingCriterion,
    PretrainedModel,
)
from paddlenlp.transformers.gpt.modeling import parallel_matmul


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
            first_key = state_dict_keys[0].split(".")
            # if use virtual pp_degree, the prefix is like 0.0.xxx
            # else it will be like 0.xxx
            use_virtual_pp_degree = first_key[0].isdigit() and first_key[1].isdigit()

            prefixs = self.get_sequential_name_prefixs()
            for k in state_dict_keys:
                name_splited = k.split(".")
                # TODO(wawltor) Fix the virtual pipeline
                if use_virtual_pp_degree:
                    idx = str(int(name_splited[0]) + int(name_splited[1]))
                    single_name = [prefixs[idx]]
                    single_name.extend(name_splited[2:])
                else:
                    idx = name_splited[0]
                    if idx == "shared_layers":
                        single_name = name_splited[2:]
                        single_name = ["gpt.embeddings"] + single_name
                    elif idx.isdigit():
                        single_name = [prefixs[idx]]
                        single_name.extend(name_splited[1:])
                    else:
                        raise ("The mapping table had bad row, please check parameter name:{}".format(k))
                mapping[".".join(single_name)] = k

            self._pipeline_name_mapping = mapping

        return self._pipeline_name_mapping

    def _prepare_pipeline_inputs_func(self, inputs):
        first_stage_keys = ["input_ids", "attention_mask"]
        last_stage_keys = ["labels"]

        def get_expected_keys(inputs, keys):
            ret = tuple([inputs.pop(k) for k in keys if k in inputs])
            if len(ret) == 1:
                ret = ret[0]
            return ret

        if type(inputs) is dict:
            return [
                get_expected_keys(inputs, first_stage_keys),
                get_expected_keys(inputs, last_stage_keys),
            ]

        keys = list(inputs[0].keys())
        inputs_batch = {key: [data.pop(key) for data in inputs] for key in keys}
        return [
            get_expected_keys(inputs_batch, first_stage_keys),
            get_expected_keys(inputs_batch, last_stage_keys),
        ]

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        assert len(self._pipeline_name_mapping) > 0, "The pipeline stage must have parameters!"
        pp_to_single_mapping = {v: k for k, v in self._pipeline_name_mapping.items()}

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            state_dict[pp_to_single_mapping[k]] = v

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

        ret = super().set_state_dict(state_dict, *args, **kwargs)
        return ret


class GPTForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """LlamaForPretraining adapted for pipeline parallelism.

    The largest change is flattening the LlamaModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    config_class = GPTConfig

    _get_tensor_parallel_mappings = GPTPretrainedModel._get_tensor_parallel_mappings
    _init_weights = GPTPretrainedModel._init_weights

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
            SharedLayerDesc("gpt", GPTEmbeddingPipe, shared_weight_attr="embedding_weight", config=config), "gpt"
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
            "gpt",
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
