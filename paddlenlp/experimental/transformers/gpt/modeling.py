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
from __future__ import annotations

import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.nn.quant import weight_quantize

from paddlenlp.experimental.transformers.fused_transformer_layers import (
    FusedMultiTransformerBase,
    FusedMultiTransformerConfig,
    FusedMultiTransformerWeightOnly,
)
from paddlenlp.experimental.transformers.generation_utils import (
    GenerationInferenceModel,
)
from paddlenlp.experimental.transformers.utils import infererence_model_from_pretrained
from paddlenlp.transformers import GPTConfig, GPTPretrainedModel
from paddlenlp.transformers.gpt.modeling import GPTEmbeddings, parallel_matmul
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import (
    dy2st_nocheck_guard_context,
    register_base_model,
)

__all__ = ["GPTInferenceModel", "GPTForCausalLMInferenceModel"]


@register_base_model
class GPTInferenceModel(GPTPretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GPTDecoderLayer`]
    Args:
        config: GPTConfig
    """

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.bos_token_id = config.bos_token_id
        self.eol_token_id = config.eol_token_id

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_layers = config.num_hidden_layers

        self.max_position_embeddings = config.max_position_embeddings

        self.embeddings = GPTEmbeddings(config)

        self.use_weight_only = False
        if config.quant_type == "weight_only_int8":
            self.use_weight_only = True
            self.quant_algo = "weight_only_int8"
        elif config.quant_type == "weight_only_int4":
            self.use_weight_only = True
            self.quant_algo = "weight_only_int4"

        # get ring_id
        ring_id = -1
        try:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            ring_id = model_parallel_group.id
        except:
            pass

        ln_scale_attrs = [
            paddle.ParamAttr(name="gpt.decoder.layers.{}.norm1.weight".format(i)) for i in range(self.num_layers)
        ]
        ln_bias_attrs = [
            paddle.ParamAttr(name="gpt.decoder.layers.{}.norm1.bias".format(i)) for i in range(self.num_layers)
        ]
        qkv_weight_attrs = [
            paddle.ParamAttr(
                name="gpt.decoder.layers.{}.self_attn.qkv_proj.weight".format(i),
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(self.num_layers)
        ]
        qkv_bias_attrs = [
            paddle.ParamAttr(name="gpt.decoder.layers.{}.self_attn.qkv_proj.bias".format(i))
            for i in range(self.num_layers)
        ]
        linear_weight_attrs = [
            paddle.ParamAttr(
                name="gpt.decoder.layers.{}.self_attn.out_proj.weight".format(i),
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(self.num_layers)
        ]
        linear_bias_attrs = [
            paddle.ParamAttr(name="gpt.decoder.layers.{}.self_attn.out_proj.bias".format(i))
            for i in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="gpt.decoder.layers.{}.norm2.weight".format(i)) for i in range(self.num_layers)
        ]
        ffn_ln_bias_attrs = [
            paddle.ParamAttr(name="gpt.decoder.layers.{}.norm2.bias".format(i)) for i in range(self.num_layers)
        ]
        ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="gpt.decoder.layers.{}.linear1.weight".format(i),
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(self.num_layers)
        ]
        ffn1_bias_attrs = [
            paddle.ParamAttr(name="gpt.decoder.layers.{}.linear1.bias".format(i)) for i in range(self.num_layers)
        ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="gpt.decoder.layers.{}.linear2.weight".format(i),
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(self.num_layers)
        ]
        ffn2_bias_attrs = [
            paddle.ParamAttr(name="gpt.decoder.layers.{}.linear2.bias".format(i)) for i in range(self.num_layers)
        ]

        qkv_weight_scale_attrs = None
        linear_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None
        if self.use_weight_only:
            qkv_weight_scale_attrs = [
                paddle.ParamAttr(name="fusemt.{}.qkv_weight_scale".format(i)) for i in range(config.n_layer)
            ]
            linear_weight_scale_attrs = [
                paddle.ParamAttr(name="fusemt.{}.linear_weight_scale".format(i)) for i in range(config.n_layer)
            ]
            ffn1_weight_scale_attrs = [
                paddle.ParamAttr(name="fusemt.{}.ffn1_weight_scale".format(i)) for i in range(config.n_layer)
            ]
            ffn2_weight_scale_attrs = [
                paddle.ParamAttr(name="fusemt.{}.ffn2_weight_scale".format(i)) for i in range(config.n_layer)
            ]

        transformer_config = FusedMultiTransformerConfig(
            config.hidden_size,
            config.num_attention_heads,
            4 * config.hidden_size,
            quant_type=config.quant_type,
            activation="gelu",
            num_layers=self.num_layers,
            nranks=config.tensor_parallel_degree,
            ring_id=ring_id,
            ln_scale_attrs=ln_scale_attrs,
            ln_bias_attrs=ln_bias_attrs,
            qkv_weight_attrs=qkv_weight_attrs,
            qkv_weight_scale_attrs=qkv_weight_scale_attrs,
            qkv_bias_attrs=qkv_bias_attrs,
            linear_weight_attrs=linear_weight_attrs,
            linear_weight_scale_attrs=linear_weight_scale_attrs,
            linear_bias_attrs=linear_bias_attrs,
            ffn_ln_scale_attrs=ffn_ln_scale_attrs,
            ffn_ln_bias_attrs=ffn_ln_bias_attrs,
            ffn1_weight_attrs=ffn1_weight_attrs,
            ffn1_weight_scale_attrs=ffn1_weight_scale_attrs,
            ffn1_bias_attrs=ffn1_bias_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_weight_scale_attrs=ffn2_weight_scale_attrs,
            ffn2_bias_attrs=ffn2_bias_attrs,
            epsilon=1e-5,
            norm_type="layernorm",
        )
        if self.use_weight_only:
            self.transformer_block = FusedMultiTransformerWeightOnly(transformer_config)
        else:
            self.transformer_block = FusedMultiTransformerBase(transformer_config)
        self.norm = nn.LayerNorm(config.hidden_size, epsilon=1e-5)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

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
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        cache=None,
        cache_kvs=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        **kwargs,
    ):
        cache = kwargs.get("cache", cache)
        is_decoder = cache is not None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if not is_decoder:
            ids_remove_padding, padding_offset, cum_offsets = self.remove_padding(input_ids, seq_len_encoder)
        else:
            ids_remove_padding = input_ids
            padding_offset = None
            cum_offsets = None

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids=ids_remove_padding, position_ids=position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        seq_lens = seq_len_decoder if is_decoder else seq_len_encoder

        hidden_states = inputs_embeds

        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                input_ids,
                hidden_states,
                cum_offsets=cum_offsets,
                padding_offset=padding_offset,
                attn_mask=paddle.cast(attention_mask, dtype=hidden_states.dtype),
                caches=cache_kvs,
                seq_lens=seq_lens,
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
            cross_attentions=None,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.transformer_block.init_weight()
        dtype = paddle.get_default_dtype()
        if "gpt.decoder.layers.0.self_attn.q_proj.weight" in state_dict.keys():
            for i in range(self.num_layers):
                q_proj_weight = state_dict.pop(f"gpt.decoder.layers.{i}.self_attn.q_proj.weight")
                k_proj_weight = state_dict.pop(f"gpt.decoder.layers.{i}.self_attn.k_proj.weight")
                v_proj_weight = state_dict.pop(f"gpt.decoder.layers.{i}.self_attn.v_proj.weight")

                q_proj_weight = q_proj_weight.transpose([1, 0]).reshape(
                    [self.num_attention_heads, self.hidden_size // self.num_attention_heads, self.hidden_size]
                )
                k_proj_weight = k_proj_weight.transpose([1, 0]).reshape(
                    [self.num_attention_heads, self.hidden_size // self.num_attention_heads, self.hidden_size]
                )
                v_proj_weight = v_proj_weight.transpose([1, 0]).reshape(
                    [self.num_attention_heads, self.hidden_size // self.num_attention_heads, self.hidden_size]
                )

                concated_qkv_weight = (
                    paddle.concat([q_proj_weight, k_proj_weight, v_proj_weight], axis=1)
                    .reshape([3 * self.hidden_size, self.hidden_size])
                    .transpose([1, 0])
                )
                state_dict[f"gpt.decoder.layers.{i}.self_attn.qkv_proj.weight"] = concated_qkv_weight

                q_proj_bias = state_dict.pop(f"gpt.decoder.layers.{i}.self_attn.q_proj.bias")
                k_proj_bias = state_dict.pop(f"gpt.decoder.layers.{i}.self_attn.k_proj.bias")
                v_proj_bias = state_dict.pop(f"gpt.decoder.layers.{i}.self_attn.v_proj.bias")

                q_proj_bias = q_proj_bias.reshape(
                    [self.num_attention_heads, self.hidden_size // self.num_attention_heads]
                )
                k_proj_bias = k_proj_bias.reshape(
                    [self.num_attention_heads, self.hidden_size // self.num_attention_heads]
                )
                v_proj_bias = v_proj_bias.reshape(
                    [self.num_attention_heads, self.hidden_size // self.num_attention_heads]
                )

                concated_qkv_bias = paddle.concat([q_proj_bias, k_proj_bias, v_proj_bias], axis=-1).reshape([-1])
                state_dict[f"gpt.decoder.layers.{i}.self_attn.qkv_proj.bias"] = concated_qkv_bias

        for k, v in state_dict.items():
            if k.startswith("gpt."):
                k = str(k.split("gpt.")[1])
            if k.find("embeddings.word_embeddings.weight") >= 0:
                self.embeddings.word_embeddings.weight.set_value(v.astype(dtype))
            elif k.find("embeddings.position_embeddings.weight") >= 0:
                self.embeddings.position_embeddings.weight.set_value(v.astype(dtype))
            elif k.find("decoder.norm.weight") >= 0:
                self.norm.weight.set_value(v.astype(dtype))
            elif k.find("decoder.norm.bias") >= 0:
                self.norm.bias.set_value(v.astype(dtype))
            else:
                if not k.startswith("decoder.layers."):
                    continue
                idx = int(k.split(".")[2])
                if k.endswith("norm1.weight"):
                    self.transformer_block.ln_scales[idx].set_value(v.astype("float32"))
                elif k.endswith("norm1.bias"):
                    self.transformer_block.ln_biases[idx].set_value(v.astype("float32"))
                elif k.endswith("self_attn.qkv_proj.weight"):
                    qkv_weight_tensor = (
                        v.reshape(
                            [
                                self.hidden_size,
                                self.num_attention_heads // self.config.tensor_parallel_degree,
                                3,
                                self.hidden_size // self.num_attention_heads,
                            ]
                        )
                        .transpose([2, 1, 3, 0])
                        .reshape(
                            [
                                -1,
                                self.hidden_size,
                            ]
                        )
                        .astype(dtype)
                    )

                    if self.use_weight_only:
                        qkv_weight_tensor = paddle.transpose(qkv_weight_tensor, perm=[1, 0])
                        qkv_quanted_weight_tensor, qkv_weight_scale_tensor = weight_quantize(
                            qkv_weight_tensor, algo=self.quant_algo
                        )
                        self.transformer_block.qkv_weights[idx].set_value(qkv_quanted_weight_tensor)
                        self.transformer_block.qkv_weights_scale[idx].set_value(qkv_weight_scale_tensor)
                    else:
                        self.transformer_block.qkv_weights[idx].set_value(qkv_weight_tensor)

                elif k.endswith("self_attn.qkv_proj.bias"):
                    self.transformer_block.qkv_biases[idx].set_value(
                        v.reshape(
                            [
                                self.num_attention_heads // self.config.tensor_parallel_degree,
                                3,
                                self.hidden_size // self.num_attention_heads,
                            ]
                        )
                        .transpose([1, 0, 2])
                        .reshape([-1])
                        .astype(dtype)
                    )
                elif k.endswith("self_attn.out_proj.weight"):
                    linear_weight_tensor = paddle.to_tensor(v.astype(dtype))
                    if self.use_weight_only:
                        linear_quanted_weight_tensor, linear_weight_scale_tensor = weight_quantize(
                            linear_weight_tensor, algo=self.quant_algo
                        )
                        self.transformer_block.linear_weights[idx].set_value(linear_quanted_weight_tensor)
                        self.transformer_block.linear_weights_scale[idx].set_value(linear_weight_scale_tensor)
                    else:
                        self.transformer_block.linear_weights[idx].set_value(linear_weight_tensor)

                elif k.endswith("self_attn.out_proj.bias"):
                    self.transformer_block.linear_biases[idx].set_value(v.astype(dtype))
                elif k.endswith("norm2.weight"):
                    self.transformer_block.ffn_ln_scales[idx].set_value(v.astype("float32"))
                elif k.endswith("norm2.bias"):
                    self.transformer_block.ffn_ln_biases[idx].set_value(v.astype("float32"))
                elif k.endswith("linear1.weight"):
                    ffn1_weight_tensor = paddle.to_tensor(v.astype(dtype))
                    if self.use_weight_only:
                        ffn1_quanted_weight_tensor, ffn1_weight_scale_tensor = weight_quantize(
                            ffn1_weight_tensor, algo=self.quant_algo
                        )
                        self.transformer_block.ffn1_weights[idx].set_value(ffn1_quanted_weight_tensor)
                        self.transformer_block.ffn1_weights_scale[idx].set_value(ffn1_weight_scale_tensor)
                    else:
                        self.transformer_block.ffn1_weights[idx].set_value(ffn1_weight_tensor)
                elif k.endswith("linear1.bias"):
                    self.transformer_block.ffn1_biases[idx].set_value(v.astype(dtype))
                elif k.endswith("linear2.weight"):
                    ffn2_weight_tensor = paddle.to_tensor(v.astype(dtype))
                    if self.use_weight_only:
                        ffn2_quanted_weight_tensor, ffn2_weight_scale_tensor = weight_quantize(
                            ffn2_weight_tensor, algo=self.quant_algo
                        )
                        self.transformer_block.ffn2_weights[idx].set_value(ffn2_quanted_weight_tensor)
                        self.transformer_block.ffn2_weights_scale[idx].set_value(ffn2_weight_scale_tensor)
                    else:
                        self.transformer_block.ffn2_weights[idx].set_value(ffn2_weight_tensor)
                elif k.endswith("linear2.bias"):
                    self.transformer_block.ffn2_biases[idx].set_value(v.astype(dtype))
                else:
                    raise ValueError("Unknow weight {}".format(k))


class GPTForCausalLMInferenceModel(GenerationInferenceModel, GPTPretrainedModel):
    """
    Dynamic Batching for GPT Model with pretraining tasks on top.
    """

    def __init__(self, config):
        super().__init__(config)
        self.gpt = GPTInferenceModel(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return infererence_model_from_pretrained(cls, pretrained_model_name_or_path, args, kwargs)

    @classmethod
    def get_cache_kvs_shape(
        cls, config: GPTConfig, max_batch_size: int = None, max_length: int = None
    ) -> list[list[int]]:
        """get cache_kvs tensor for gpt model

        Args:
            max_batch_size (int): the max batch size
            max_length (int | None, optional): the max_length of cache_kvs. Defaults to None.

        Returns:
            list[paddle.Tensor]: the list tensor shape for cache
        """
        if max_length is None:
            max_length = config.max_position_embeddings

        cache_kvs = []
        for _ in range(config.num_hidden_layers):
            cache_kvs.append(
                [
                    2,
                    max_batch_size,
                    config.num_attention_heads // max(config.tensor_parallel_degree, 1),
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
        if cache is not None:
            input_ids = tgt_ids
            position_ids = tgt_pos
            attention_mask = (tgt_generation_mask - 1) * 1e4
        else:
            attention_mask = (attention_mask - 1) * 1e4

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "cache_kvs": cache_kvs,
            "seq_len_encoder": seq_len_encoder,
            "seq_len_decoder": seq_len_decoder,
            "cache": cache,
        }
        return model_inputs

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id
        ).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype("int64")
        else:
            attention_mask = paddle.ones_like(input_ids, dtype="int64")
        return paddle.unsqueeze(attention_mask, axis=[1, 2])

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        cache=None,
        cache_kvs=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache=cache,
            cache_kvs=cache_kvs,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = parallel_matmul(
            hidden_states, self.gpt.embeddings.word_embeddings.weight, tensor_parallel_output=False
        )

        if not return_dict:
            return (logits, outputs[1:])

        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.gpt.set_state_dict({k: state_dict[k] for k in state_dict.keys()})
