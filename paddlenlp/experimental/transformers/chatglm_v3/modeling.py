# encoding=utf-8
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

import numpy as np
import paddle
from paddle import nn
from paddle.distributed import fleet

from paddlenlp.experimental.transformers.fused_transformer_layers import (
    FusedBlockMultiTransformer,
    FusedMultiTransformerBase,
    FusedMultiTransformerConfig,
)
from paddlenlp.experimental.transformers.generation_utils import (
    GenerationBlockInferenceModel,
)

from paddlenlp.transformers.conversion_utils import split_param_func
from paddlenlp.transformers.llama.modeling import LlamaLMHead
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)

from paddlenlp.transformers.model_utils import (
    dy2st_nocheck_guard_context,
    register_base_model,
)
from paddlenlp.utils.log import logger

from paddlenlp.transformers.chatglm_v3.configuration import ChatGLMv3Config
from paddlenlp.transformers.chatglm_v3.modeling import ChatGLMv3PretrainedModel


def gqa_2_mha_fused():
    def fn(fused_param, split_nums=2, is_qkv=False, num_heads=None, num_key_value_heads=None):
        """split function for splitting weights

        (1) fuse_attention_qkv
            fused weight => [q1,k1,v1,q2,k2,v2,q3,k3,v3,q4,k4,v4]
                 or for GQA [q1,q2,k1,v1,q3,q4,k2,v2]
            after split
            q => [q1,q2,q3,q4]
            k => [k1,k2,k3,k4] or [k1,k2] for GQA
            v => [v1,v2,v3,v4] or [v1,v2] for GQA
        (2) fuse_attention_ffn
            directly split weight to 2 parts
            [gate_weight, up_weight] => [gate_weight], [up_weight]

        Args:
            fused_param (_type_): len(fused_param)=1, only one weight to be splitted
            split_nums (int, optional): split_nums. Defaults to 2.
            is_qkv (bool, optional): for attention qkv weights. Defaults to False.
            num_heads (_type_, optional): query heads. Defaults to None.
            num_key_value_heads (_type_, optional): key and value heads. Defaults to None.

        Returns:
            _type_: splitted weights
        """
        concat_fn = np.concatenate
        split_fn = np.split
        if isinstance(fused_param, paddle.Tensor):
            concat_fn = paddle.concat
            split_fn = paddle.split

        if is_qkv:
            # fuse_attention_qkv
            assert num_heads, f"num_heads should be number of heads for Q, but got {num_heads}"
            assert (
                num_key_value_heads
            ), f"num_key_value_heads should be number of key_value_heads for K and V, but got {num_key_value_heads}"
            num_query_heads_per_groups = num_heads // num_key_value_heads

            q_list, k_list, v_list = [], [], []
            split_heads = split_fn(fused_param, num_heads + 2 * num_key_value_heads, axis=-1)

            for i in range(num_key_value_heads):
                q_list += split_heads[
                          i * (num_query_heads_per_groups + 2): (i + 1) * (num_query_heads_per_groups + 2) - 2]
                k_list.extend([split_heads[(i + 1) * (num_query_heads_per_groups + 2) - 2] for j in
                               range(num_query_heads_per_groups)])
                v_list.extend([split_heads[(i + 1) * (num_query_heads_per_groups + 2) - 1] for j in
                               range(num_query_heads_per_groups)])
            return concat_fn(q_list, axis=-1), concat_fn(k_list, axis=-1), concat_fn(v_list, axis=-1)
        else:
            # fuse_attention_ffn
            return split_fn(fused_param, split_nums, axis=-1)

    return fn

def gqa_2_mha():
    def fn(fused_param, split_nums=2, is_qkv=False, num_heads=None, num_key_value_heads=None):
        """split function for splitting weights

        (1) fuse_attention_qkv
            fused weight => [q1,q2, q3, q4, k1, k2, k3, k4, v1, v2, v3, v4]
                 or for GQA [q1,q2,q3,q4, k1,k2,v1,v2]
            after split
            q => [q1,q2,q3,q4]
            k => [k1,k2,k3,k4] or [k1,k2] for GQA
            v => [v1,v2,v3,v4] or [v1,v2] for GQA
        (2) fuse_attention_ffn
            directly split weight to 2 parts
            [gate_weight, up_weight] => [gate_weight], [up_weight]

        Args:
            fused_param (_type_): len(fused_param)=1, only one weight to be splitted
            split_nums (int, optional): split_nums. Defaults to 2.
            is_qkv (bool, optional): for attention qkv weights. Defaults to False.
            num_heads (_type_, optional): query heads. Defaults to None.
            num_key_value_heads (_type_, optional): key and value heads. Defaults to None.

        Returns:
            _type_: splitted weights
        """
        concat_fn = np.concatenate
        split_fn = np.split
        if isinstance(fused_param, paddle.Tensor):
            concat_fn = paddle.concat
            split_fn = paddle.split

        if is_qkv:
            # fuse_attention_qkv
            assert num_heads, f"num_heads should be number of heads for Q, but got {num_heads}"
            assert (
                num_key_value_heads
            ), f"num_key_value_heads should be number of key_value_heads for K and V, but got {num_key_value_heads}"
            num_query_heads_per_groups = num_heads // num_key_value_heads

            q_list, k_list, v_list = [], [], []
            split_heads = split_fn(fused_param, num_heads + 2 * num_key_value_heads, axis=-1)
            q_list = split_heads[:num_heads]

            k_list.extend([split_heads[num_heads + i] for i in range(num_key_value_heads) for j in
                           range(num_query_heads_per_groups)])
            v_list.extend([split_heads[num_heads + num_key_value_heads + i] for i in range(num_key_value_heads) for j in
                           range(num_query_heads_per_groups)])

            return concat_fn(q_list, axis=-1), concat_fn(k_list, axis=-1), concat_fn(v_list, axis=-1)
        else:
            # fuse_attention_ffn
            return split_fn(fused_param, split_nums, axis=-1)

    return fn


class FusedLlamaRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states):
        result = paddle.incubate.nn.functional.fused_rms_norm(
            hidden_states, self.weight, None, self.variance_epsilon, begin_norm_axis=1
        )
        if isinstance(result, tuple):
            return result[0]
        return result



@register_base_model
class ChatGLMv3InferenceModel(ChatGLMv3PretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config: ChatGLMv3Config
    """

    def __init__(self, config: ChatGLMv3Config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.num_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.epsilon = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings
        self.quant_type = config.quant_type

        self.use_weight_only = False

        if self.quant_type is not None and "weight_only_int" in self.quant_type:
            self.use_weight_only = True
        elif self.quant_type is not None and "a8w8" in self.quant_type:
            self.quant_model_path = config.model_name_or_path
            self.shift = config.quantization_config.shift
            self.smooth = config.quantization_config.smooth
            self.shift_smooth_all_linears = config.quantization_config.shift_smooth_all_linears
        else:
            self.use_weight_only = False

        if self.use_weight_only:
            assert (
                    self.quant_type == "weight_only_int8" or self.quant_type == "weight_only_int4"
            ), "Expected quant_type equal to 'weight_only_int8' or 'weight_only_int4', but received {}".format(
                self.quant_type
            )

        if config.tensor_parallel_degree > 1:
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        # get ring_id
        ring_id = -1
        try:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            ring_id = model_parallel_group.id
        except:
            pass

        ln_scale_attrs = [paddle.ParamAttr(name="fusechatglmv3.{}.ln_scale".format(i)) for i in range(self.num_layers)]
        qkv_weight_attrs = [
            paddle.ParamAttr(
                name="fusechatglmv3.{}.qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        qkv_bias_attrs = [
            paddle.ParamAttr(
                name="fusechatglmv3.{}.qkv_bias".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        out_proj_weight_attrs = [
            paddle.ParamAttr(
                name="fusechatglmv3.{}.out_proj_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="fusechatglmv3.{}.ffn_ln_scale".format(i)) for i in range(self.num_layers)
        ]
        ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="fusechatglmv3.{}.ffn1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fusechatglmv3.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        qkv_out_scale_attrs = None
        linear_out_scale_attrs = None
        ffn1_out_scale_attrs = None
        ffn2_out_scale_attrs = None
        linear_shift_attrs = None
        linear_smooth_attrs = None
        ffn2_shift_attrs = None
        ffn2_smooth_attrs = None
        ln_bias_attrs = None

        out_proj_bias_attrs = None
        ffn_ln_bias_attrs = None
        ffn1_bias_attrs = None
        ffn2_bias_attrs = None
        qkv_weight_scale_attrs = None
        out_proj_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None
        cache_k_scale_attrs = None
        cache_v_scale_attrs = None
        cache_k_out_scale_attrs = None
        cache_v_out_scale_attrs = None
        transformer_config = FusedMultiTransformerConfig(
            self.hidden_size,
            self.num_attention_heads,
            self.intermediate_size,
            activation="swiglu",
            num_layers=config.num_hidden_layers,
            nranks=config.tensor_parallel_degree,
            ring_id=ring_id,
            ln_scale_attrs=ln_scale_attrs,
            qkv_weight_attrs=qkv_weight_attrs,
            qkv_weight_scale_attrs=qkv_weight_scale_attrs,
            linear_weight_attrs=out_proj_weight_attrs,
            linear_weight_scale_attrs=out_proj_weight_scale_attrs,
            ffn_ln_scale_attrs=ffn_ln_scale_attrs,
            ffn1_weight_attrs=ffn1_weight_attrs,
            ffn1_weight_scale_attrs=ffn1_weight_scale_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_weight_scale_attrs=ffn2_weight_scale_attrs,
            qkv_out_scale_attrs=qkv_out_scale_attrs,
            linear_out_scale_attrs=linear_out_scale_attrs,
            ffn1_out_scale_attrs=ffn1_out_scale_attrs,
            ffn2_out_scale_attrs=ffn2_out_scale_attrs,
            linear_shift_attrs=linear_shift_attrs,
            linear_smooth_attrs=linear_smooth_attrs,
            ffn2_shift_attrs=ffn2_shift_attrs,
            ffn2_smooth_attrs=ffn2_smooth_attrs,
            ln_bias_attrs=ln_bias_attrs,
            qkv_bias_attrs=qkv_bias_attrs,
            linear_bias_attrs=out_proj_bias_attrs,
            ffn_ln_bias_attrs=ffn_ln_bias_attrs,
            ffn1_bias_attrs=ffn1_bias_attrs,
            ffn2_bias_attrs=ffn2_bias_attrs,
            cache_k_scale_attrs=cache_k_scale_attrs,
            cache_v_scale_attrs=cache_v_scale_attrs,
            cache_k_out_scale_attrs=cache_k_out_scale_attrs,
            cache_v_out_scale_attrs=cache_v_out_scale_attrs,
            epsilon=self.epsilon,
            norm_type="rmsnorm",
            use_neox_rotary_style=True,
            rank_id=config.tensor_parallel_rank,
        )

        self.set_transformer_block(transformer_config)
        self.norm = FusedLlamaRMSNorm(config)

        self.cache_kvs = None
        self.head_dim_shape_tensor = paddle.ones((self.hidden_size // self.num_attention_heads), dtype="int8")

        self.gradient_checkpointing = False

    def set_transformer_block(self, transformer_config):
        self.transformer_block = FusedMultiTransformerBase(transformer_config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def remove_padding(self, input_ids, seq_lens_this_time):
        cum_offsets_now = paddle.cumsum(paddle.max(seq_lens_this_time) - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        from paddlenlp_ops import get_padding_offset
        ids_remove_padding, cum_offsets, padding_offset = get_padding_offset(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time
        )
        return ids_remove_padding, padding_offset, cum_offsets

    # This function is a little different from prepare_input_ids_for_generation in paddlenlp/transformers/generation/utils.py
    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            use_cache=None,
            cache_kvs=None,
            pre_caches=None,
            seq_len_encoder=None,
            seq_len_decoder=None,
            past_key_values=None,
            output_attentions=False,
            output_hidden_states=None,
            return_dict=False,
            **kwargs,
    ):
        # kwargs["cache"] is used used to distinguish between encoder and decoder phase.
        past_key_values = kwargs.get("cache", None)
        is_decoder = past_key_values is not None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # genereate a fake input_ids according to inputs_embeds
        # this is usually occurred in img2txt multimodal model when first enter into this forward function.
        if input_ids is None and inputs_embeds is not None:
            input_ids = self.prepare_input_ids_for_generation(self.config.bos_token_id, inputs_embeds)
        if inputs_embeds is not None:
            batch, seq_len, hidden_dim = inputs_embeds.shape
            # merge batch and seq_len dimension.
            inputs_embeds = inputs_embeds.reshape([batch * seq_len, hidden_dim])

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        cache_kvs = cache_kvs if cache_kvs is not None else self.cache_kvs

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is None:
            past_key_values = tuple([None] * self.config.num_hidden_layers)

        if not is_decoder:
            ids_remove_padding, padding_offset, cum_offsets = self.remove_padding(input_ids, seq_len_encoder)
        else:
            ids_remove_padding = input_ids.squeeze(axis=1)
            padding_offset = None
            cum_offsets = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(ids_remove_padding)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        seq_lens = seq_len_decoder if is_decoder else seq_len_encoder

        position_offset = 0
        theta = 10000.0
        if not is_decoder and pre_caches is not None:
            position_offset = 128

        from paddlenlp_ops import fused_get_rotary_embedding

        new_rope = fused_get_rotary_embedding(
            input_ids, position_ids, self.head_dim_shape_tensor, position_offset, theta, True
        )

        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                input_ids,
                hidden_states,
                cum_offsets=cum_offsets,
                padding_offset=padding_offset,
                attn_mask=paddle.cast(attention_mask, dtype=hidden_states.dtype),
                caches=cache_kvs,
                pre_caches=pre_caches,
                pre_caches_length=position_offset,
                seq_lens=seq_lens,
                rotary_embs=new_rope,
                rotary_emb_dims=1,
                time_step=paddle.increment(paddle.shape(attention_mask)[-1], -1) if is_decoder else None,
            )
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        unfused_state_dict = {}
        split_fn = split_param_func()

        self.embed_tokens.weight.set_value(
            paddle.to_tensor(state_dict["transformer.word_embeddings.weight"]).cast(self.embed_tokens.weight.dtype)
        )
        self.norm.weight.set_value(
            paddle.to_tensor(state_dict["transformer.final_layernorm.weight"]).cast(self.norm.weight.dtype))

        for idx in range(self.config.num_hidden_layers):
            logger.info(f"set state for layer {idx}")

            if self.use_weight_only:
                logger.info("weight only is enabled")

            if "transformer.layers.{}.self_attention.query_key_value.weight".format(idx) in state_dict.keys():
                concated_qkv_weight = np.concatenate(
                    gqa_2_mha_fused()(
                        state_dict["transformer.layers.{}.self_attention.query_key_value.weight".format(idx)],
                        is_qkv=True,
                        num_heads=self.num_attention_heads // self.config.tensor_parallel_degree,
                        num_key_value_heads=self.num_key_value_heads // self.config.tensor_parallel_degree,
                    ),
                    axis=-1,
                ).transpose(1, 0)

                concated_qkv_bias = np.concatenate(
                    gqa_2_mha_fused()(
                        state_dict["transformer.layers.{}.self_attention.query_key_value.bias".format(idx)],
                        is_qkv=True,
                        num_heads=self.num_attention_heads // self.config.tensor_parallel_degree,
                        num_key_value_heads=self.num_key_value_heads // self.config.tensor_parallel_degree,
                    ),
                    axis=-1,
                )

            else:
                unfused_state_dict = {}
                unfused_state_dict_bias = {}
                unfused_state_dict["self_attn.q_proj.weight"] = state_dict[
                    "transformer.layers.{}.self_attention.q_proj.weight".format(idx)
                ]
                unfused_state_dict_bias["self_attn.q_proj.bias"] = state_dict[
                    "transformer.layers.{}.self_attention.q_proj.bias".format(idx)
                ]
                unfused_state_dict["self_attn.k_proj.weight"] = state_dict[
                    "transformer.layers.{}.self_attention.k_proj.weight".format(idx)
                ]
                unfused_state_dict_bias["self_attn.k_proj.bias"] = state_dict[
                    "transformer.layers.{}.self_attention.k_proj.bias".format(idx)
                ]
                unfused_state_dict["self_attn.v_proj.weight"] = state_dict[
                    "transformer.layers.{}.self_attention.v_proj.weight".format(idx)
                ]

                unfused_state_dict_bias["self_attn.v_proj.bias"] = state_dict[
                    "transformer.layers.{}.self_attention.v_proj.bias".format(idx)
                ]

                qkv_weight = np.concatenate(
                    [
                        unfused_state_dict["self_attn.q_proj.weight"],
                        unfused_state_dict["self_attn.k_proj.weight"],
                        unfused_state_dict["self_attn.v_proj.weight"],
                    ],
                    axis=-1,
                )

                qkv_biases = np.concatenate(
                    [
                        unfused_state_dict_bias["self_attn.q_proj.bias"],
                        unfused_state_dict_bias["self_attn.k_proj.bias"],
                        unfused_state_dict_bias["self_attn.v_proj.bias"],
                    ],
                    axis=-1,
                )

                concated_qkv_weight = np.concatenate(
                    gqa_2_mha()(
                        qkv_weight,
                        is_qkv=True,
                        num_heads=self.num_attention_heads // self.config.tensor_parallel_degree,
                        num_key_value_heads=self.num_key_value_heads // self.config.tensor_parallel_degree,
                    ),
                    axis=-1,
                ).transpose(1, 0)

                concated_qkv_bias = np.concatenate(
                    gqa_2_mha()(
                        qkv_biases,
                        is_qkv=True,
                        num_heads=self.num_attention_heads // self.config.tensor_parallel_degree,
                        num_key_value_heads=self.num_key_value_heads // self.config.tensor_parallel_degree,
                    ),
                    axis=-1,
                )

            if "transformer.layers.{}.mlp.dense_h_to_4h.weight".format(idx) in state_dict.keys():
                concated_ffn1_weight = np.concatenate(
                    split_fn(state_dict["transformer.layers.{}.mlp.dense_h_to_4h.weight".format(idx)]), axis=-1
                )
            else:
                unfused_state_dict["mlp.gate_proj.weight"] = state_dict[
                    "transformer.layers.{}.mlp.gate_proj.weight".format(idx)
                ]
                unfused_state_dict["mlp.up_proj.weight"] = state_dict[
                    "transformer.layers.{}.mlp.up_proj.weight".format(idx)]
                concated_ffn1_weight = np.concatenate(
                    [unfused_state_dict["mlp.gate_proj.weight"], unfused_state_dict["mlp.up_proj.weight"]], axis=-1
                )

            ffn1_weight_tensor = paddle.to_tensor(concated_ffn1_weight)

            qkv_weight_tensor = paddle.to_tensor(concated_qkv_weight)
            qkv_biases_tensor = paddle.to_tensor(concated_qkv_bias)

            self.transformer_block.qkv_weights[idx].set_value(
                qkv_weight_tensor.cast(self.transformer_block.qkv_weights[idx].dtype)
            )

            self.transformer_block.qkv_biases[idx].set_value(
                qkv_biases_tensor.cast(self.transformer_block.qkv_biases[idx].dtype)
            )

            linear_weight_tensor = paddle.to_tensor(
                state_dict["transformer.layers.{}.self_attention.dense.weight".format(idx)])

            self.transformer_block.linear_weights[idx].set_value(
                linear_weight_tensor.cast(self.transformer_block.linear_weights[idx].dtype)
            )

            self.transformer_block.ffn1_weights[idx].set_value(
                ffn1_weight_tensor.cast(self.transformer_block.ffn1_weights[idx].dtype)
            )

            ffn2_weight_tensor = paddle.to_tensor(
                state_dict["transformer.layers.{}.mlp.dense_4h_to_h.weight".format(idx)])

            self.transformer_block.ffn2_weights[idx].set_value(
                ffn2_weight_tensor.cast(self.transformer_block.ffn2_weights[idx].dtype)
            )

            self.transformer_block.ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["transformer.layers.{}.input_layernorm.weight".format(idx)]).cast(
                    self.transformer_block.ln_scales[idx].dtype
                )
            )

            self.transformer_block.ffn_ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["transformer.layers.{}.post_attention_layernorm.weight".format(idx)]).cast(
                    self.transformer_block.ffn_ln_scales[idx].dtype
                )
            )


@register_base_model
class ChatGLMv3BlockInferenceModel(ChatGLMv3InferenceModel):

    def __init__(self, config: ChatGLMv3Config):
        super().__init__(config)
        self.max_seq_len = config.max_seq_len
        self.block_size = config.block_size

    def set_transformer_block(self, transformer_config):
        self.transformer_block = FusedBlockMultiTransformer(transformer_config)

    def remove_padding(self, input_ids, seq_lens_this_time):
        cum_offsets_now = paddle.cumsum(self.max_seq_len - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        from paddlenlp_ops import get_padding_offset_v2

        ids_remove_padding, cum_offsets, padding_offset, cu_seqlens_q, cu_seqlens_k = get_padding_offset_v2(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time
        )
        return ids_remove_padding, padding_offset, cum_offsets, cu_seqlens_q, cu_seqlens_k

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            caches=None,
            pre_caches=None,
            output_attentions=False,
            output_hidden_states=None,
            return_dict=False,
            **kwargs,
    ):
        seq_lens_this_time = kwargs.get("seq_lens_this_time", None)
        rope_emb = kwargs.get("rope_emb", None)
        ids_remove_padding, padding_offset, cum_offsets, cu_seqlens_q, cu_seqlens_k = self.remove_padding(
            input_ids, seq_lens_this_time
        )
        kwargs["cu_seqlens_q"] = cu_seqlens_q
        kwargs["cu_seqlens_k"] = cu_seqlens_k
        kwargs["padding_offsets"] = padding_offset
        kwargs["max_input_length"] = self.max_seq_len

        inputs_embeds = self.embed_tokens(ids_remove_padding)

        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                input_ids=input_ids,
                src=inputs_embeds,
                cum_offsets=cum_offsets,
                attn_mask=attention_mask,
                caches=caches,
                pre_caches=pre_caches,
                rotary_embs=rope_emb,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class ChatGLMv3ForCausalLMBlockInferenceModel(GenerationBlockInferenceModel, ChatGLMv3PretrainedModel):
    """
    Dynamic Batching for LLaMA Model with pretraining tasks on top.
    """

    _keys_to_ignore_on_load_missing = [r"transformer.output_layer.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = ChatGLMv3BlockInferenceModel(config)
        self.lm_head = LlamaLMHead(config)

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: ChatGLMv3Config, is_split=True):

        logger.info("ChatGLMv3 inference model _get_tensor_parallel_mappings")

        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                "output_layer.weight": partial(fn, is_column=True),
                # Row Linear
                "transformer.word_embeddings.weight": partial(fn, is_column=False),
                "transformer.layers.0.self_attention.dense.weight": partial(fn, is_column=False),
                "transformer.layers.0.mlp.dense_4h_to_h.weight": partial(fn, is_column=False),
            }

            # Column Linear
            if config.fuse_attention_qkv:
                base_actions["transformer.layers.0.self_attention.query_key_value.weight"] = partial(fn, is_column=True)
                base_actions["transformer.layers.0.self_attention.query_key_value.bias"] = partial(fn, is_column=True)
            else:
                base_actions["transformer.layers.0.self_attention.q_proj.weight"] = partial(fn, is_column=True)
                base_actions["transformer.layers.0.self_attention.q_proj.bias"] = partial(fn, is_column=True)
                # if we have enough num_key_value_heads to split, then split it.
                if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                    base_actions["transformer.layers.0.self_attention.k_proj.weight"] = partial(fn, is_column=True)
                    base_actions["transformer.layers.0.self_attention.v_proj.weight"] = partial(fn, is_column=True)

                    base_actions["transformer.layers.0.self_attention.k_proj.bias"] = partial(fn, is_column=True)
                    base_actions["transformer.layers.0.self_attention.v_proj.bias"] = partial(fn, is_column=True)
            if config.fuse_attention_ffn:
                base_actions["transformer.layers.0.mlp.dense_h_to_4h.weight"] = partial(
                    fn, is_column=True, is_naive_2fuse=True
                )
            else:
                base_actions["transformer.layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
                base_actions["transformer.layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "transformer.layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("transformer.layers.0.", f"transformer.layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # TODO: Support safetensors loading.
        kwargs["use_safetensors"] = False
        from paddlenlp.transformers.utils import (
            ContextManagers,
            is_safetensors_available,
        )

        from_hf_hub = kwargs.pop("from_hf_hub", False)
        config = kwargs.pop("config", None)
        from_aistudio = kwargs.get("from_aistudio", False)
        subfolder = kwargs.get("subfolder", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)
        convert_from_torch = kwargs.pop("convert_from_torch", None)
        cache_dir = kwargs.pop("cache_dir", None)

        init_contexts = []
        with ContextManagers(init_contexts):
            model = cls(config)

        if not config.single_card_ptq:
            resolved_archive_file = pretrained_model_name_or_path
        else:
            resolved_archive_file = cls._resolve_model_file_path(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                subfolder=subfolder,
                from_hf_hub=from_hf_hub,
                from_aistudio=from_aistudio,
                config=config,
                convert_from_torch=convert_from_torch,
                use_safetensors=use_safetensors,
                variant=variant,
            )[0]
        logger.info(f"Load model form {resolved_archive_file}")

        if config.tensor_parallel_degree > 1 and config.single_card_ptq:
            logger.info(f"convert_tensor_parallel {config.tensor_parallel_degree}")
            model.state_dict = model.convert_tensor_parallel(resolved_archive_file, config)
        elif config.tensor_parallel_degree > 1:
            resolved_archive_file = os.path.join(
                resolved_archive_file, f"mp_{config.tensor_parallel_rank:0>2d}_sharding_00_pp_00", "model.pdparams"
            )
            model.state_dict = paddle.load(resolved_archive_file, return_numpy=True)
        else:
            model.state_dict = paddle.load(resolved_archive_file, return_numpy=True)

        model.set_state_dict(model.state_dict)

        return model

    @classmethod
    def get_cache_kvs_shape(
            cls, config: ChatGLMv3Config, max_batch_size: int = None, max_length: int = None
    ) -> list[list[int]]:
        """get cache_kvs tensor for llama model

        Args:
            max_batch_size (int): the max batch size
            max_length (int | None, optional): the max_length of cache_kvs. Defaults to None.

        Returns:
            list[paddle.Tensor]: the list tensor shape for cache
        """
        max_block_per_seq = (config.max_seq_len + config.block_size - 1) // config.block_size
        if max_batch_size == -1:
            max_block_nums = None
        else:
            max_block_nums = max_batch_size * max_block_per_seq

        cache_kvs = []
        for _ in range(config.num_hidden_layers):
            cache_kv_shape = [
                max_block_nums,

                config.num_attention_heads // max(config.tensor_parallel_degree, 1),
                config.block_size,
                config.hidden_size // config.num_attention_heads,
            ]
            cache_kvs.append(cache_kv_shape)
            cache_kvs.append(cache_kv_shape)
        return cache_kvs

    def prepare_inputs_for_generation(self, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        input_ids = kwargs["input_ids"]
        src_mask = kwargs.get("src_mask", None)
        block_tables = kwargs.get("block_tables", None)

        pre_caches = kwargs.get("pre_caches", None)
        caches = kwargs.get("caches", None)

        rope_emb = kwargs["rope_emb"]
        seq_lens_this_time = kwargs["seq_lens_this_time"]
        seq_lens_encoder = kwargs["seq_lens_encoder"]
        seq_lens_decoder = kwargs["seq_lens_decoder"]
        k_quant_scales = kwargs.get("k_quant_scales", None)
        v_quant_scales = kwargs.get("v_quant_scales", None)
        k_dequant_scales = kwargs.get("k_dequant_scales", None)
        v_dequant_scales = kwargs.get("v_dequant_scales", None)
        model_inputs = {
            "input_ids": input_ids,
            "src_mask": src_mask,
            "rope_emb": rope_emb,
            "pre_caches": pre_caches,
            "caches": caches,
            "seq_lens_this_time": seq_lens_this_time,
            "seq_lens_encoder": seq_lens_encoder,
            "seq_lens_decoder": seq_lens_decoder,
            "block_tables": block_tables,
            "k_quant_scales": k_quant_scales,
            "v_quant_scales": v_quant_scales,
            "k_dequant_scales": k_dequant_scales,
            "v_dequant_scales": v_dequant_scales,
        }
        return model_inputs

    def forward(
            self,
            input_ids,
            src_mask=None,
            pre_caches=None,
            caches=None,
            seq_lens_this_time=None,
            seq_lens_encoder=None,
            seq_lens_decoder=None,
            rope_emb=None,
            block_tables=None,
            k_quant_scales=None,
            v_quant_scales=None,
            k_dequant_scales=None,
            v_dequant_scales=None,
    ):
        outputs = self.transformer(
            input_ids,
            src_mask=src_mask,
            caches=caches,
            rope_emb=rope_emb,
            block_tables=block_tables,
            pre_caches=pre_caches,
            seq_lens_this_time=seq_lens_this_time,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            k_quant_scales=k_quant_scales,
            v_quant_scales=v_quant_scales,
            k_dequant_scales=k_dequant_scales,
            v_dequant_scales=v_dequant_scales,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(
            hidden_states,
            tensor_parallel_output=False,
        )

        return logits

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        # if "output_layer.weight" in state_dict:
        self.lm_head.weight.set_value(
            paddle.to_tensor(state_dict["output_layer.weight"]).cast(self.lm_head.weight.dtype)
        )
        self.transformer.set_state_dict({k: state_dict[k] for k in state_dict.keys()})