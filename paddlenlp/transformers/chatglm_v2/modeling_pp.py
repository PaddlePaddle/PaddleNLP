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
import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp
except:
    pass

from paddlenlp.transformers.model_utils import PipelinePretrainedModel

from .modeling import (
    ChatGLMv2Config,
    Chatglmv2LMHead,
    ChatGLMv2PretrainedModel,
    ChatGLMv2PretrainingCriterion,
    Embedding,
    GLMBlock,
    RMSNorm,
)

__all__ = ["ChatGLMv2ForCausalLMPipe"]


def get_hcg():
    return fleet.get_hybrid_communicate_group()


def get_attr(layer, name):
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


def parse_args(args):
    if isinstance(args, tuple):
        if len(args) == 6:
            hidden_states, attention_mask, position_ids, rotary_pos_emb, kv_cache, use_cache = args
        elif len(args) == 5:
            hidden_states, attention_mask, position_ids, rotary_pos_emb, kv_cache = args
            use_cache = None
        elif len(args) == 4:
            hidden_states, attention_mask, position_ids, rotary_pos_emb = args
            kv_cache = None
            use_cache = None
        elif len(args) == 3:
            hidden_states, attention_mask, position_ids = args
            rotary_pos_emb = None
            kv_cache = None
            use_cache = None
        elif len(args) == 2:
            hidden_states, attention_mask = args
            position_ids = None
            rotary_pos_emb = None
            kv_cache = None
            use_cache = None
    else:
        hidden_states = args
        attention_mask, position_ids, rotary_pos_emb, kv_cache, use_cache = None, None, None, None, None

    if position_ids is not None:
        position_ids.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    if rotary_pos_emb is not None:
        rotary_pos_emb.stop_gradient = True

    if kv_cache is not None:
        kv_cache.stop_gradient = True

    if use_cache is not None:
        use_cache.stop_gradient = True

    return hidden_states, attention_mask, position_ids, rotary_pos_emb, kv_cache, use_cache


def return_args(
    hidden_states, attention_mask=None, position_ids=None, rotary_pos_emb=None, kv_cache=None, use_cache=None
):
    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if position_ids is not None:
        ret += (position_ids.clone(),)
    if rotary_pos_emb is not None:
        ret += (rotary_pos_emb.clone(),)
    if kv_cache is not None:
        ret += (kv_cache.clone(),)
    if use_cache is not None:
        ret += (use_cache.clone(),)

    if len(ret) == 1:
        ret = ret[0]

    return ret


def forward_impl(self, seq_len: int, n_elem: int, base: int = 10000):
    """Enhanced Transformer with Rotary Position Embedding.
    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (paddle.arange(0, n_elem, 2, dtype="float32") / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = paddle.arange(0, seq_len, dtype=theta.dtype)

    # Calculate the product of position index and $\theta_i$
    idx_theta = paddle.outer(seq_idx, theta).astype(self.default_dtype)

    cache = paddle.stack([paddle.cos(idx_theta), paddle.sin(idx_theta)], axis=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if self.default_dtype in (paddle.float16, paddle.bfloat16, paddle.int8):
        cache = cache.astype(self.default_dtype)
        # cache = cache.bfloat16() if dtype == paddle.bfloat16 else cache.astype("float16")
    return cache


class EmbeddingPipe(Embedding):
    """Extends Embedding to forward attention_mask through the pipeline."""

    def __init__(self, config: ChatGLMv2Config):
        super().__init__(config)
        self.default_dtype = paddle.get_default_dtype()

    @property
    def embedding_weight(self):
        return get_attr(self.word_embeddings, "weight")

    def forward(self, args):
        input_ids, attention_mask, position_ids, rotary_pos_emb, kv_cache, use_cache = parse_args(args)
        input_ids.stop_gradient = True
        inputs_embeds = super().forward(input_ids=input_ids)
        batch_size, seq_length = input_ids.shape

        if self.config.sequence_parallel:
            seq_length, batch_size, hidden_size = inputs_embeds.shape
            inputs_embeds = paddle.reshape_(inputs_embeds, [batch_size * seq_length, hidden_size])
            # [seq_len * bs / n, num_head * head_dim] (n is mp parallelism)
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        if attention_mask is None:
            attention_mask = paddle.ones((batch_size, 1, seq_length, seq_length), dtype="bool")
        if len(attention_mask.shape) == 2:
            # from Tokenizer
            attention_mask = (
                attention_mask.unsqueeze(axis=[1, 2]).expand([batch_size, 1, seq_length, seq_length]).astype("bool")
            )
        elif len(attention_mask.shape) == 3:
            # [batch_size,tgt_length, src_length] -> [batch_size, 1, tgt_length, src_length]
            attention_mask = attention_mask.unsqueeze(1).astype("bool")
        elif len(attention_mask.shape) == 4:
            attention_mask = attention_mask.astype("bool")

        causal_mask = paddle.tril(paddle.ones([batch_size, 1, seq_length, seq_length])).astype("bool")
        attention_mask = attention_mask & causal_mask
        zero = paddle.zeros(attention_mask.shape, dtype=inputs_embeds.dtype)
        neg_inf = paddle.full_like(attention_mask, paddle.finfo(inputs_embeds.dtype).min, dtype=inputs_embeds.dtype)
        attention_mask = paddle.where(attention_mask, zero, neg_inf)
        # Rotary positional embeddings
        self.max_sequence_length = self.config.max_sequence_length
        rotary_dim = (
            self.config.hidden_size // self.config.num_attention_heads
            if self.config.kv_channels is None
            else self.config.kv_channels
        )
        rotary_pos_emb = forward_impl(self, self.max_sequence_length, rotary_dim // 2)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]

        rotary_pos_emb = rotary_pos_emb.transpose([1, 0, 2, 3])

        return return_args(inputs_embeds, attention_mask, position_ids, rotary_pos_emb, kv_cache, use_cache)


class GLMBlockPipe(GLMBlock):
    """Extends GLMBlock to forward attention_mask through the pipeline."""

    def forward(self, args):
        hidden_states, attention_mask, position_ids, rotary_pos_emb, kv_cache, use_cache = parse_args(args)
        hidden_states, kv_cache = super().forward(hidden_states, attention_mask, rotary_pos_emb, kv_cache, use_cache)
        return return_args(hidden_states, attention_mask, position_ids, rotary_pos_emb, kv_cache, use_cache)


class RMSNormPipe(RMSNorm):
    def forward(self, args):
        hidden_states, attention_mask, position_ids, rotary_pos_emb, kv_cache, use_cache = parse_args(args)
        hidden_states = super().forward(hidden_states)
        return hidden_states


class Chatglmv2LMHeadPipe(Chatglmv2LMHead):
    def __init__(self, config):
        super(Chatglmv2LMHeadPipe, self).__init__(config)


class ChatGLMv2ForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """ChatGLMv2ForPretraining adapted for pipeline parallelism.

    The largest change is flattening the ChatGLMv2Model class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    config_class = ChatGLMv2Config

    get_masks = ChatGLMv2PretrainedModel.get_masks
    _get_tensor_parallel_mappings = ChatGLMv2PretrainedModel._get_tensor_parallel_mappings
    init_weights = ChatGLMv2PretrainedModel.init_weights
    get_position_ids = ChatGLMv2PretrainedModel.get_position_ids
    _get_name_mappings = ChatGLMv2PretrainedModel._get_name_mappings

    # NO base_model_prefix !!!!

    def __init__(self, config):
        self.config = config

        virtual_pp_degree = getattr(self.config, "virtual_pp_degree", 1)

        hcg = get_hcg()
        tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)

        config.tensor_parallel_degree = tensor_parallel_degree
        config.tensor_parallel_rank = tensor_parallel_rank

        self.add_sequential_layer(
            LayerDesc(EmbeddingPipe, config=config),
            "embedding",
        )
        for i in range(config.num_hidden_layers):
            self.add_sequential_layer(
                LayerDesc(GLMBlockPipe, config=config, layer_number=i),
                f"encoder.layers.{i}",
            )

        self.add_sequential_layer(
            LayerDesc(RMSNormPipe, config=config),
            "encoder.final_layernorm",
        )
        self.add_sequential_layer(
            LayerDesc(Chatglmv2LMHeadPipe, config=config),
            "output_layer",
        )

        recompute_interval = 0
        # if self.config.recompute and recompute_granularity == "full":
        #    assert pp_recompute_interval <= config.num_hidden_layers // (
        #        virtual_pp_degree * get_hcg().topology().get_dim_size("pipe")
        #    ), "pp recompute interval should smaller than num layers of each pp chunk"
        #    recompute_interval = pp_recompute_interval

        seg_method = "layer:GLMBlock"
        if config.num_hidden_layers % get_hcg().topology().get_dim_size("pipe") != 0:
            seg_method = "uniform"

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=ChatGLMv2PretrainingCriterion(config),
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
