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
from paddlenlp.transformers.bloom.modeling import (
    BloomBlock,
    BloomConfig,
    BloomLMHead,
    BloomModel,
    BloomPretrainedModel,
    BloomPretrainingCriterion,
    build_alibi_tensor,
)


def get_hcg():
    return fleet.get_hybrid_communicate_group()


def parse_args(args):
    if isinstance(args, tuple):
        if len(args) == 3:
            hidden_states, attention_mask, alibi = args
        elif len(args) == 2:
            hidden_states, attention_mask = args
            alibi = None
    else:
        hidden_states = args
        attention_mask, alibi = None, None

    if alibi is not None:
        alibi.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    return hidden_states, attention_mask, alibi


def return_args(hidden_states, attention_mask=None, alibi=None):
    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if alibi is not None:
        ret += (alibi.clone(),)
    if len(ret) == 1:
        ret = ret[0]

    return ret


class BloomEmbeddingPipe(nn.Layer):
    """Extends BloomEmbeddings to forward attention_mask through the pipeline."""

    _prepare_attn_mask = BloomModel._prepare_attn_mask

    def __init__(self, config):
        super(BloomEmbeddingPipe, self).__init__()
        self.config = config
        if config.tensor_parallel_degree > 1:
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, args):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_ids, attention_mask, alibi = parse_args(args)

        input_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_ids.shape
        if attention_mask is not None:
            attention_mask = BloomModel._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), 0, input_embeds.dtype
            )
            attention_mask.stop_gradient = True
        else:
            attention_mask = paddle.ones([batch_size, seq_length], dtype=paddle.get_default_dtype())
            alibi = build_alibi_tensor(attention_mask, self.config.n_head, dtype=input_embeds.dtype)

            causal_mask = self._prepare_attn_mask(
                attention_mask,
                input_shape=(batch_size, seq_length),
                past_key_values_length=0,
            )

            if self.config.tensor_parallel_degree > 1:
                block_size = self.config.n_head // self.config.tensor_parallel_degree
                alibi = alibi[
                    :,
                    self.config.tensor_parallel_rank
                    * block_size : (self.config.tensor_parallel_rank + 1)
                    * block_size,
                ]
                alibi = alibi.reshape([batch_size * block_size, 1, seq_length])
                # causal_mask = paddle.cast(
                #     paddle.repeat_interleave(paddle.cast(causal_mask, "int32"), block_size, axis=0), "bool"
                # )
                causal_mask = paddle.repeat_interleave(paddle.cast(causal_mask, "int32"), block_size, axis=0)
            else:
                alibi = alibi.reshape([batch_size * self.config.n_head, 1, seq_length])
                # causal_mask = paddle.cast(
                #     paddle.repeat_interleave(paddle.cast(causal_mask, "int32"), self.config.n_head, axis=0), "bool"
                # )
                paddle.repeat_interleave(paddle.cast(causal_mask, "int32"), self.config.n_head, axis=0)

            alibi.stop_gradient = True
            causal_mask.stop_gradient = True

        return return_args(input_embeds, causal_mask, alibi)


class BloomBlockPipe(BloomBlock):
    _prepare_attn_mask = BloomModel._prepare_attn_mask

    def forward(self, args):
        hidden_states, attention_mask, alibi = parse_args(args)
        # use int32 instead of bool since pp not support bool
        causal_mask = paddle.cast(attention_mask, "bool")
        hidden_states = super().forward(hidden_states, attention_mask=causal_mask, alibi=alibi)
        return return_args(hidden_states[0], attention_mask, alibi)


class LayerNormPipe(nn.LayerNorm):
    def __init__(self, config):
        self.config = config
        super().__init__(config.hidden_size, epsilon=config.layer_norm_epsilon)

    def forward(self, args):
        hidden_states, attention_mask, alibi = parse_args(args)
        return super().forward(hidden_states)


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
                if use_virtual_pp_degree:
                    idx = str(int(name_splited[0]) + int(name_splited[1]))
                    single_name = [prefixs[idx]]
                    single_name.extend(name_splited[2:])
                else:
                    idx = name_splited[0]
                    single_name = [prefixs[idx]]
                    single_name.extend(name_splited[1:])
                mapping[".".join(single_name)] = k

            self._pipeline_name_mapping = mapping

        return self._pipeline_name_mapping

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


class BloomForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """BloomForPretraining adapted for pipeline parallelism.

    The largest change is flattening the BloomModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    config_class = BloomConfig

    _get_tensor_parallel_mappings = BloomPretrainedModel._get_tensor_parallel_mappings
    _init_weights = BloomPretrainedModel._init_weights

    # NO base_model_prefix !!!!

    def __init__(
        self,
        config,
        # use_recompute=None,
        # scale_qk_by_layer_num=True,
        # recompute_granularity="full",
        # virtual_pp_degree=4,
        # sequence_parallel=False,
        # no_recompute_layers=None,
        pp_recompute_interval=1,
    ):
        self.config = config

        use_recompute = self.config.use_recompute
        recompute_granularity = self.config.recompute_granularity
        virtual_pp_degree = self.config.virtual_pp_degree

        hcg = get_hcg()
        tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)

        config.tensor_parallel_degree = tensor_parallel_degree
        config.tensor_parallel_rank = tensor_parallel_rank

        self.add_sequential_layer(LayerDesc(BloomEmbeddingPipe, config=config), "bloom")
        for i in range(config.n_layer):
            self.add_sequential_layer(LayerDesc(BloomBlockPipe, config=config), f"bloom.h.{i}")

        self.add_sequential_layer(LayerDesc(LayerNormPipe, config=config), "bloom.ln_f")
        self.add_sequential_layer(LayerDesc(BloomLMHead, config=config), "lm_head")

        recompute_interval = 0
        if use_recompute and recompute_granularity == "full":
            assert pp_recompute_interval <= config.n_layer // (
                virtual_pp_degree * get_hcg().topology().get_dim_size("pipe")
            ), "pp recompute interval should smaller than num layers of each pp chunk"
            recompute_interval = pp_recompute_interval

        seg_method = "layer:BloomBlock"
        if config.n_layer % get_hcg().topology().get_dim_size("pipe") != 0:
            seg_method = "uniform"

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=BloomPretrainingCriterion(config),
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
