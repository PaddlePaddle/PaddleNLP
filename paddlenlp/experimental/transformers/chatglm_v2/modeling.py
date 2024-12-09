# Copyright (c) 2023 ChatGLM2-6B Model Team and PaddlePaddle Authors. All Rights Reserved.
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


from typing import Optional

import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.nn.quant import weight_quantize

from paddlenlp.experimental.transformers.fused_transformer_layers import (
    FusedBlockMultiTransformer,
    FusedBlockMultiTransformerWeightOnly,
    FusedMultiTransformerBase,
    FusedMultiTransformerConfig,
    FusedMultiTransformerWeightOnly,
)
from paddlenlp.experimental.transformers.generation_utils import (
    GenerationBlockInferenceModel,
    GenerationInferenceModel,
)
from paddlenlp.experimental.transformers.utils import infererence_model_from_pretrained
from paddlenlp.transformers import ChatGLMv2Config, ChatGLMv2PretrainedModel
from paddlenlp.transformers.chatglm_v2.modeling import (
    Embedding,
    LayerNorm,
    RMSNorm,
    RotaryEmbedding,
)
from paddlenlp.transformers.model_utils import (
    dy2st_nocheck_guard_context,
    register_base_model,
)

__all__ = ["ChatGLMv2ForCausalLMInferenceModel", "ChatGLMv2ForCausalLMBlockInferenceModel"]


@register_base_model
class ChatGLMv2InferenceModel(ChatGLMv2PretrainedModel):
    def __init__(self, config: ChatGLMv2Config, empty_init=True):
        super().__init__(config)
        self.embedding = Embedding(config)

        # Rotary positional embeddings
        self.max_sequence_length = config.max_sequence_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2)

        if config.tensor_parallel_degree > 1:
            if config.tensor_parallel_degree > 2:
                raise ValueError(
                    "ChatGLM2 does not support `tensor_parallel_degree` > 2. Consider using Sharding stage 3"
                )
            self.output_layer = fleet.meta_parallel.ColumnParallelLinear(
                config.hidden_size,
                config.padded_vocab_size,
                has_bias=False,
                gather_output=not config.tensor_parallel_output,
            )
        else:
            self.output_layer = nn.Linear(config.hidden_size, config.padded_vocab_size, bias_attr=False)

        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_heads
        self.multi_query_group_num = config.multi_query_group_num

        self.use_weight_only = False
        if config.quant_type == "weight_only_int8":
            self.use_weight_only = True
            self.quant_algo = "weight_only_int8"
        elif config.quant_type == "weight_only_int4":
            self.use_weight_only = True
            self.quant_algo = "weight_only_int4"

        ln_scale_attrs = [
            paddle.ParamAttr(name="encoder.layers.{}.input_layernorm.weight".format(i))
            for i in range(config.num_hidden_layers)
        ]

        qkv_weight_attrs = [
            paddle.ParamAttr(
                name="encoder.layers.{}.qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(config.num_hidden_layers)
        ]
        qkv_bias_attrs = [
            paddle.ParamAttr(name="encoder.layers.{}.qkv_bias".format(i)) for i in range(config.num_hidden_layers)
        ]

        out_proj_weight_attrs = [
            paddle.ParamAttr(
                name="encoder.layers.{}.self_attention.dense.weight".format(i),
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(config.num_hidden_layers)
        ]

        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="encoder.layers.{}.post_attention_layernorm.weight".format(i))
            for i in range(config.num_hidden_layers)
        ]

        ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="encoder.layers.{}.mlp.dense_h_to_4h.weight".format(i),
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(config.num_hidden_layers)
        ]

        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="encoder.layers.{}.mlp.dense_4h_to_h.weight".format(i),
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(config.num_hidden_layers)
        ]

        qkv_weight_scale_attrs = None
        out_proj_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None
        if self.use_weight_only:
            qkv_weight_scale_attrs = [
                paddle.ParamAttr(name="encoder.layers.{}.qkv_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            out_proj_weight_scale_attrs = [
                paddle.ParamAttr(name="encoder.layers.{}.self_attention.dense.weight_scale".format(i))
                for i in range(self.num_layers)
            ]
            ffn1_weight_scale_attrs = [
                paddle.ParamAttr(name="encoder.layers.{}.mlp.dense_h_to_4h.weight_scale".format(i))
                for i in range(self.num_layers)
            ]
            ffn2_weight_scale_attrs = [
                paddle.ParamAttr(name="encoder.layers.{}.mlp.dense_4h_to_h.weight_scale".format(i))
                for i in range(self.num_layers)
            ]
        transformer_config = FusedMultiTransformerConfig(
            config.hidden_size,
            config.num_attention_heads,
            config.ffn_hidden_size,
            dropout_rate=0.0,
            quant_type=config.quant_type,
            activation="swiglu",
            normalize_before=True,
            num_layers=config.num_hidden_layers,
            nranks=1,
            ring_id=-1,
            ln_scale_attrs=ln_scale_attrs,
            qkv_weight_attrs=qkv_weight_attrs,
            qkv_weight_scale_attrs=qkv_weight_scale_attrs,
            qkv_bias_attrs=qkv_bias_attrs,
            linear_weight_attrs=out_proj_weight_attrs,
            linear_weight_scale_attrs=out_proj_weight_scale_attrs,
            ffn_ln_scale_attrs=ffn_ln_scale_attrs,
            ffn1_weight_attrs=ffn1_weight_attrs,
            ffn1_weight_scale_attrs=ffn1_weight_scale_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_weight_scale_attrs=ffn2_weight_scale_attrs,
            epsilon=config.layernorm_epsilon,
            norm_type="rmsnorm",
            kv_num_heads=config.multi_query_group_num,
        )

        self.set_transformer_block(transformer_config)

        self.post_layer_norm = config.post_layer_norm
        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config)

    def set_transformer_block(self, transformer_config):
        if self.use_weight_only:
            self.transformer_block = FusedMultiTransformerWeightOnly(transformer_config)
        else:
            self.transformer_block = FusedMultiTransformerBase(transformer_config)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

    def remove_padding(self, input_ids, seq_lens_this_time):
        cum_offsets_now = paddle.cumsum(paddle.max(seq_lens_this_time) - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        from paddlenlp_ops import get_padding_offset

        ids_remove_padding, cum_offsets, padding_offset = get_padding_offset(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time
        )
        return ids_remove_padding, padding_offset, cum_offsets

    def forward(
        self,
        input_ids=None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds=None,
        use_cache=None,
        cache_kvs=None,
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

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not is_decoder:
            ids_remove_padding, padding_offset, cum_offsets = self.remove_padding(input_ids, seq_len_encoder)
        else:
            ids_remove_padding = input_ids
            padding_offset = None
            cum_offsets = None

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding.word_embeddings(ids_remove_padding)
        hidden_states = inputs_embeds

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.max_sequence_length)

        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
            rotary_pos_emb = rotary_pos_emb[:, :seq_length, :, :]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]

        ones = paddle.ones([batch_size, seq_length, self.head_size // 4], dtype=paddle.get_default_dtype())
        zeros = paddle.zeros([batch_size, seq_length, self.head_size // 4], dtype=paddle.get_default_dtype())
        # make it to be [2, batch, seq_len, rotary_dim]
        rotary_pos_emb = rotary_pos_emb.transpose([3, 0, 1, 2])
        # The following code is for consistency with PaddleNLP/csrc/generation/encode_rotary_qk.cu, so boring.
        cos = rotary_pos_emb[0]
        sin = rotary_pos_emb[1]
        cos = paddle.concat([cos, ones], axis=-1)
        sin = paddle.concat([sin, zeros], axis=-1)
        rotary_pos_emb = paddle.stack([cos, sin], axis=0)
        rotary_pos_emb = (
            rotary_pos_emb.unsqueeze(-1).tile([1, 1, 1, 1, 2]).reshape([2, batch_size, seq_length, self.head_size])
        )

        # Run encoder.
        seq_lens = seq_len_decoder if is_decoder else seq_len_encoder
        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                input_ids,
                hidden_states,
                cum_offsets=cum_offsets,
                padding_offset=padding_offset,
                attn_mask=paddle.cast(attention_mask, dtype=hidden_states.dtype),
                caches=cache_kvs,
                pre_caches=None,
                pre_caches_length=0,
                seq_lens=seq_lens,
                rotary_embs=paddle.cast(rotary_pos_emb, "float32"),
                rotary_emb_dims=1,
                time_step=paddle.increment(paddle.shape(attention_mask)[-1], -1) if is_decoder else None,
            )

        hidden_states = self.final_layernorm(hidden_states)

        return tuple(v for v in [hidden_states, None, None, None] if v is not None)

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.transformer_block.init_weight()

        # find the real name.
        def key(name):
            result_list = []
            for i in state_dict.keys():
                if i.find(name) >= 0:
                    result_list.append(i)
            assert len(result_list) == 1, name + " must be only one in state_dict"
            return result_list[0]

        self.embedding.word_embeddings.weight.set_value(state_dict.pop(key("embedding.word_embeddings.weight")))
        self.final_layernorm.weight.set_value(state_dict.pop(key("encoder.final_layernorm.weight")))
        self.output_layer.weight.set_value(state_dict.pop(key("output_layer.weight")))

        for i in range(self.num_layers):
            ln_scale = state_dict.pop(key("encoder.layers.{}.input_layernorm.weight".format(i)))

            concated_qkv_weight = state_dict.pop(
                key("encoder.layers.{}.self_attention.query_key_value.weight".format(i))
            )
            concated_qkv_weight = concated_qkv_weight.transpose([1, 0])
            concated_qkv_weight = paddle.to_tensor(concated_qkv_weight)

            concated_qkv_bias = state_dict.pop(key("encoder.layers.{}.self_attention.query_key_value.bias".format(i)))
            concated_qkv_bias = paddle.to_tensor(concated_qkv_bias)

            out_proj_weight = state_dict.pop(key("encoder.layers.{}.self_attention.dense.weight".format(i)))

            ffn_ln_scale = state_dict.pop(key("encoder.layers.{}.post_attention_layernorm.weight".format(i)))

            ffn1_weight = state_dict.pop(key("encoder.layers.{}.mlp.dense_h_to_4h.weight".format(i)))
            ffn2_weight = state_dict.pop(key("encoder.layers.{}.mlp.dense_4h_to_h.weight".format(i)))

            self.transformer_block.ln_scales[i].set_value(ln_scale)

            if self.use_weight_only:
                qkv_weight_tensor = paddle.to_tensor(concated_qkv_weight)
                qkv_weight_tensor = paddle.transpose(qkv_weight_tensor, perm=[1, 0])
                qkv_quanted_weight_tensor, qkv_weight_scale_tensor = weight_quantize(
                    qkv_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.qkv_weights[i].set_value(qkv_quanted_weight_tensor)
                self.transformer_block.qkv_weights_scale[i].set_value(qkv_weight_scale_tensor)
            else:
                self.transformer_block.qkv_weights[i].set_value(concated_qkv_weight)

            self.transformer_block.qkv_biases[i].set_value(concated_qkv_bias)

            if self.use_weight_only:
                linear_quanted_weight_tensor, linear_weight_scale_tensor = weight_quantize(
                    paddle.to_tensor(out_proj_weight), algo=self.quant_algo
                )
                self.transformer_block.linear_weights[i].set_value(linear_quanted_weight_tensor)
                self.transformer_block.linear_weights_scale[i].set_value(linear_weight_scale_tensor)
            else:
                self.transformer_block.linear_weights[i].set_value(out_proj_weight)

            self.transformer_block.ffn_ln_scales[i].set_value(ffn_ln_scale)

            if self.use_weight_only:
                ffn1_quanted_weight_tensor, ffn1_weight_scale_tensor = weight_quantize(
                    paddle.to_tensor(ffn1_weight), algo=self.quant_algo
                )
                self.transformer_block.ffn1_weights[i].set_value(ffn1_quanted_weight_tensor)
                self.transformer_block.ffn1_weights_scale[i].set_value(ffn1_weight_scale_tensor)
            else:
                self.transformer_block.ffn1_weights[i].set_value(ffn1_weight)

            if self.use_weight_only:
                ffn2_quanted_weight_tensor, ffn2_weight_scale_tensor = weight_quantize(
                    paddle.to_tensor(ffn2_weight), algo=self.quant_algo
                )
                self.transformer_block.ffn2_weights[i].set_value(ffn2_quanted_weight_tensor)
                self.transformer_block.ffn2_weights_scale[i].set_value(ffn2_weight_scale_tensor)
            else:
                self.transformer_block.ffn2_weights[i].set_value(ffn2_weight)


@register_base_model
class ChatGLMv2BlockInferenceModel(ChatGLMv2InferenceModel):
    def __init__(self, config: ChatGLMv2Config):
        super().__init__(config)
        self.max_seq_len = config.max_sequence_length
        self.block_size = config.block_size

    def set_transformer_block(self, transformer_config):
        if self.use_weight_only:
            self.transformer_block = FusedBlockMultiTransformerWeightOnly(transformer_config)
        else:
            self.transformer_block = FusedBlockMultiTransformer(transformer_config)

    def remove_padding(self, input_ids, seq_lens_this_time, draft_tokens=None, seq_lens_encoder=None):
        cum_offsets_now = paddle.cumsum(self.max_seq_len - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        from paddlenlp_ops import get_padding_offset_v2

        ids_remove_padding, cum_offsets, padding_offset, cu_seqlens_q, cu_seqlens_k = get_padding_offset_v2(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time, draft_tokens, seq_lens_encoder
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

        inputs_embeds = self.embedding.word_embeddings(ids_remove_padding)

        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                input_ids=input_ids,
                src=inputs_embeds,
                cum_offsets=cum_offsets,
                attn_mask=attention_mask,
                caches=caches,
                pre_caches=None,
                rotary_embs=rope_emb,
                **kwargs,
            )
        hidden_states = self.final_layernorm(hidden_states)

        return tuple(v for v in [hidden_states, None, None, None] if v is not None)


class ChatGLMv2ForCausalLMInferenceModel(GenerationInferenceModel, ChatGLMv2PretrainedModel):
    def __init__(self, config: ChatGLMv2Config):
        super().__init__(config)
        self.max_sequence_length = config.max_sequence_length
        self.chatglm_v2 = ChatGLMv2InferenceModel(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return infererence_model_from_pretrained(cls, pretrained_model_name_or_path, args, kwargs)

    @classmethod
    def get_cache_kvs_shape(cls, config: ChatGLMv2Config, max_batch_size: int = None, max_length: int = None):
        """get cache_kvs tensor for opt model

        Args:
            max_batch_size (int): the max batch size
            max_length (int | None, optional): the max_length of cache_kvs. Defaults to None.

        Returns:
            list[paddle.Tensor]: the list tensor shape for cache
        """

        if max_length is None:
            max_length = config.max_sequence_length

        cache_kvs = []
        for _ in range(config.num_hidden_layers):
            cache_kvs.append(
                [
                    2,
                    max_batch_size,
                    config.multi_query_group_num,
                    max_length,
                    config.hidden_size // config.num_attention_heads,
                ]
            )
        return cache_kvs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        cache_kvs,
        seq_len_encoder,
        seq_len_decoder,
        tgt_ids,
        tgt_pos,
        tgt_generation_mask,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        cache = kwargs.get("cache", None)
        pre_caches = kwargs.get("pre_caches", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if cache is not None:
            input_ids = tgt_ids
            position_ids = tgt_pos
            attention_mask = (tgt_generation_mask - 1) * 1e4
            # make inputs_embeds be none in decoder phase.
            # in forward function, it will be assigned according to input_ids.
            inputs_embeds = None
        else:
            attention_mask = (attention_mask - 1) * 1e4
        model_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "cache_kvs": cache_kvs,
            "seq_len_encoder": seq_len_encoder,
            "seq_len_decoder": seq_len_decoder,
            "cache": cache,
            "pre_caches": pre_caches,
        }
        return model_inputs

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        cache=None,
        cache_kvs=None,
        pre_caches=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.chatglm_v2(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache=cache,
            cache_kvs=cache_kvs,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.chatglm_v2.output_layer(hidden_states)
        output = (lm_logits,) + transformer_outputs[1:]
        return output

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.chatglm_v2.set_state_dict(state_dict)


class ChatGLMv2ForCausalLMBlockInferenceModel(GenerationBlockInferenceModel, ChatGLMv2PretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.chatglm_v2 = ChatGLMv2BlockInferenceModel(config)
        self.max_sequence_length = config.max_sequence_length

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return infererence_model_from_pretrained(cls, pretrained_model_name_or_path, args, kwargs)

    @classmethod
    def get_cache_kvs_shape(cls, config: ChatGLMv2Config, max_batch_size: int = None, max_length: int = None):
        """get cache_kvs tensor for chatglmv2 model

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
                config.multi_query_group_num,
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
        outputs = self.chatglm_v2(
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
        lm_logits = self.chatglm_v2.output_layer(hidden_states)
        output = (lm_logits,) + outputs[1:]

        return output

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.chatglm_v2.set_state_dict(state_dict)
