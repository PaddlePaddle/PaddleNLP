# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import paddle
from paddle import nn
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
from paddlenlp.transformers import QWenConfig, QWenPretrainedModel
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from paddlenlp.transformers.model_utils import (
    dy2st_nocheck_guard_context,
    register_base_model,
)
from paddlenlp.transformers.qwen.modeling import QWenLMHead, QWenPretrainingCriterion

__all__ = ["QWenForCausalLMInferenceModel", "QWenForQWenVLInferenceModel"]


class FusedQWenRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.eps = config.layer_norm_epsilon
        self.weight = paddle.create_parameter(
            shape=[config.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, x):
        return paddle.incubate.nn.functional.fused_rms_norm(x, self.weight, None, self.eps, begin_norm_axis=1)[0]


@register_base_model
class QWenInferenceModel(QWenPretrainedModel):
    def __init__(self, config: QWenConfig):
        super(QWenPretrainedModel, self).__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.num_layers = config.num_hidden_layers
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.max_position_embeddings = config.max_position_embeddings
        self.quant_type = config.quant_type

        self.use_weight_only = False
        if config.quant_type == "weight_only_int8":
            self.use_weight_only = True
            self.quant_algo = "weight_only_int8"
        elif config.quant_type == "weight_only_int4":
            self.use_weight_only = True
            self.quant_algo = "weight_only_int4"

        if self.use_weight_only:
            assert (
                self.quant_algo == "weight_only_int8" or self.quant_algo == "weight_only_int4"
            ), "Expected quant_type equal to 'weight_only_int8' or 'weight_only_int4', but received {}".format(
                self.quant_algo
            )

        self.wte = nn.Embedding(self.vocab_size, self.hidden_size)

        ln_scale_attrs = [paddle.ParamAttr(name="fuseqwen.{}.ln_scale".format(i)) for i in range(self.num_layers)]
        qkv_weight_attrs = [
            paddle.ParamAttr(
                name="fuseqwen.{}.qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        qkv_bias_attrs = [paddle.ParamAttr(name="fuseqwen.{}.qkv_bias".format(i)) for i in range(self.num_layers)]
        out_proj_weight_attrs = [
            paddle.ParamAttr(
                name="fuseqwen.{}.out_proj_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="fuseqwen.{}.ffn_ln_scale".format(i)) for i in range(self.num_layers)
        ]
        ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="fuseqwen.{}.ffn1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fuseqwen.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        qkv_weight_scale_attrs = None
        out_proj_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None

        if self.use_weight_only:
            qkv_weight_scale_attrs = [
                paddle.ParamAttr(name="fuseqwen.{}.qkv_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            out_proj_weight_scale_attrs = [
                paddle.ParamAttr(name="fuseqwen.{}.out_proj_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn1_weight_scale_attrs = [
                paddle.ParamAttr(name="fuseqwen.{}.ffn1_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn2_weight_scale_attrs = [
                paddle.ParamAttr(name="fuseqwen.{}.ffn2_weight_scale".format(i)) for i in range(self.num_layers)
            ]

        transformer_config = FusedMultiTransformerConfig(
            self.hidden_size,
            self.num_attention_heads,
            self.intermediate_size // 2,
            quant_type=self.quant_type,
            activation="swiglu",
            num_layers=config.num_hidden_layers,
            nranks=1,
            ring_id=-1,
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
            qkv_bias_attrs=qkv_bias_attrs,
            epsilon=self.layer_norm_epsilon,
            norm_type="rmsnorm",
            use_neox_rotary_style=True,
        )

        if self.use_weight_only:
            self.transformer_block = FusedMultiTransformerWeightOnly(transformer_config)
        else:
            self.transformer_block = FusedMultiTransformerBase(transformer_config)

        self.ln_f = FusedQWenRMSNorm(config)

        self.cache_kvs = None
        self.head_dim_shape_tensor = paddle.ones((self.hidden_size // self.num_attention_heads), dtype="int8")

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.transformer_block.init_weight()
        dtype = paddle.get_default_dtype()
        wte_weight = paddle.to_tensor(state_dict["qwen.wte.weight"], dtype=dtype)
        ln_f_weight = paddle.to_tensor(state_dict["qwen.ln_f.weight"], dtype=self.ln_f.weight.dtype)
        self.wte.weight.set_value(wte_weight)
        self.ln_f.weight.set_value(ln_f_weight)

        for idx in range(self.num_layers):
            ln_scale = paddle.to_tensor(
                state_dict["qwen.h.{}.ln_1.weight".format(idx)], dtype=self.transformer_block.ln_scales[idx].dtype
            )
            self.transformer_block.ln_scales[idx].set_value(ln_scale)

            qkv_weight = paddle.to_tensor(
                state_dict["qwen.h.{}.attn.c_attn.weight".format(idx)].transpose([1, 0]), dtype=dtype
            )
            if self.use_weight_only:
                qkv_weight = paddle.transpose(qkv_weight, perm=[1, 0])
                qkv_quanted_weight, qkv_weight_scale = weight_quantize(qkv_weight, algo=self.quant_algo)
                self.transformer_block.qkv_weights[idx].set_value(qkv_quanted_weight)
                self.transformer_block.qkv_weights_scale[idx].set_value(qkv_weight_scale)
            else:
                self.transformer_block.qkv_weights[idx].set_value(qkv_weight)

            qkv_bias = paddle.to_tensor(state_dict["qwen.h.{}.attn.c_attn.bias".format(idx)], dtype=dtype)
            self.transformer_block.qkv_biases[idx].set_value(qkv_bias)

            linear_weight = paddle.to_tensor(state_dict["qwen.h.{}.attn.c_proj.weight".format(idx)], dtype=dtype)
            if self.use_weight_only:
                linear_quanted_weight, linear_weight_scale = weight_quantize(linear_weight, algo=self.quant_algo)
                self.transformer_block.linear_weights[idx].set_value(linear_quanted_weight)
                self.transformer_block.linear_weights_scale[idx].set_value(linear_weight_scale)
            else:
                self.transformer_block.linear_weights[idx].set_value(linear_weight)

            ffn_ln_scale = paddle.to_tensor(
                state_dict["qwen.h.{}.ln_2.weight".format(idx)], dtype=self.transformer_block.ffn_ln_scales[idx].dtype
            )
            self.transformer_block.ffn_ln_scales[idx].set_value(ffn_ln_scale)

            up_weight = paddle.to_tensor(state_dict["qwen.h.{}.mlp.w1.weight".format(idx)], dtype=dtype)
            gate_weight = paddle.to_tensor(state_dict["qwen.h.{}.mlp.w2.weight".format(idx)], dtype=dtype)
            ffn1_weight = paddle.concat(x=[gate_weight, up_weight], axis=-1)
            if self.use_weight_only:
                ffn1_quanted_weight, ffn1_weight_scale = weight_quantize(ffn1_weight, algo=self.quant_algo)
                self.transformer_block.ffn1_weights[idx].set_value(ffn1_quanted_weight)
                self.transformer_block.ffn1_weights_scale[idx].set_value(ffn1_weight_scale)
            else:
                self.transformer_block.ffn1_weights[idx].set_value(ffn1_weight)

            ffn2_weight = paddle.to_tensor(state_dict["qwen.h.{}.mlp.c_proj.weight".format(idx)], dtype=dtype)
            if self.use_weight_only:
                ffn2_quanted_weight, ffn2_weight_scale = weight_quantize(ffn2_weight, algo=self.quant_algo)
                self.transformer_block.ffn2_weights[idx].set_value(ffn2_quanted_weight)
                self.transformer_block.ffn2_weights_scale[idx].set_value(ffn2_weight_scale)
            else:
                self.transformer_block.ffn2_weights[idx].set_value(ffn2_weight)

    def remove_padding(self, input_ids, seq_lens_this_time):
        cum_offsets_now = paddle.cumsum(paddle.max(seq_lens_this_time) - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        from paddlenlp_ops import get_padding_offset

        ids_remove_padding, cum_offsets, padding_offset = get_padding_offset(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time
        )
        return ids_remove_padding, padding_offset, cum_offsets

    # This function is a little different from prepare_input_ids_for_generation in paddlenlp/transformers/generation/utils.py,
    # it is used to generate fake input_ids according to inputs_embeds length.
    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.full([batch_size, seq_len], bos_token_id, dtype="int64")

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

        # generate a fake input_ids according to inputs_embeds
        # this is usually occurred in img2txt multimodal model when first enter into this forward function.
        if input_ids is None and inputs_embeds is not None:
            input_ids = self.prepare_input_ids_for_generation(self.config.bos_token_id, inputs_embeds)
        if inputs_embeds is not None:
            batch, seq_len, hidden_dim = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape([batch * seq_len, hidden_dim])

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is None:
            past_key_values = tuple([None] * self.config.num_hidden_layers)

        if not is_decoder:
            ids_remove_padding, padding_offset, cum_offsets = self.remove_padding(input_ids, seq_len_encoder)
        else:
            ids_remove_padding = input_ids
            padding_offset = None
            cum_offsets = None

        if inputs_embeds is None:
            inputs_embeds = self.wte(ids_remove_padding)
        hidden_states = inputs_embeds

        # decoder layers
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

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

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class QWenForCausalLMInferenceModel(GenerationInferenceModel, QWenPretrainedModel):
    def __init__(self, config: QWenConfig, **kwargs):
        super(QWenForCausalLMInferenceModel, self).__init__(config)
        self.qwen = QWenInferenceModel(config)
        self.lm_head = QWenLMHead(config)
        self.criterion = QWenPretrainingCriterion(config)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return infererence_model_from_pretrained(cls, pretrained_model_name_or_path, args, kwargs)

    @classmethod
    def get_cache_kvs_shape(
        cls, config: QWenConfig, max_batch_size: int = None, max_length: int = None
    ) -> list[list[int]]:
        """get cache_kvs tensor for qwen model

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
        input_ids,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        cache=None,
        cache_kvs=None,
        pre_caches=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.qwen(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache=cache,
            cache_kvs=cache_kvs,
            pre_caches=pre_caches,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is togather with ParallelCrossEntropy
        tensor_parallel_output = (
            self.config.tensor_parallel_output and labels is not None and self.config.tensor_parallel_degree > 1
        )
        lm_logits = self.lm_head(hidden_states, tensor_parallel_output=tensor_parallel_output)

        loss = None
        if labels is not None:
            loss = self.criterion(lm_logits, labels)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        if "lm_head.weight" in state_dict:
            lm_head_weight = paddle.to_tensor(state_dict["lm_head.weight"], dtype=self.lm_head.weight.dtype)
            self.lm_head.weight.set_value(lm_head_weight)
        self.qwen.set_state_dict({k: state_dict[k] for k in state_dict.keys()})


class QWenForQWenVLInferenceModel(QWenForCausalLMInferenceModel):
    """
    This class is 99% like QWenForCausalLMInferenceModel.
    Used only for QWenVL's second part.
    """

    # This function corresponds to QWenVL's second part, only used for QWenVL.
    @paddle.no_grad()
    def generate_text_with_image_features(
        self,
        input_ids: paddle.Tensor,
        image_features: paddle.Tensor,
        img_pos: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_ids=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        min_length=None,
        max_length=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        step_idx=None,
        stop_flags=None,
        tgt_ids=None,
        tgt_pos=None,
        tgt_generation_mask=None,
        pre_ids=None,
        stop_nums=None,
        cache_kvs=[],
        inputs_embeds=None,
        **generate_kwargs
    ) -> paddle.Tensor:
        inputs_embeds = self.qwen.wte(input_ids)
        inputs_embeds_dtype = inputs_embeds.dtype
        if inputs_embeds_dtype != paddle.float32:
            inputs_embeds = paddle.cast(inputs_embeds, paddle.float32)
            image_features = paddle.cast(image_features, paddle.float32)

        for idx, (i, image_start_idx, image_end_idx) in enumerate(img_pos):
            index = paddle.arange(image_start_idx + 1, image_end_idx).unsqueeze(-1)
            inputs_embeds[i] = paddle.scatter(inputs_embeds[i], index, image_features[idx])

        if inputs_embeds_dtype != paddle.float32:
            inputs_embeds = paddle.cast(inputs_embeds, inputs_embeds_dtype)

        outputs = self.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            penalty_score=penalty_score,
            frequency_score=frequency_score,
            presence_score=presence_score,
            min_length=min_length,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            step_idx=step_idx,
            stop_flags=stop_flags,
            tgt_ids=tgt_ids,
            tgt_pos=tgt_pos,
            tgt_generation_mask=tgt_generation_mask,
            pre_ids=pre_ids,
            stop_nums=stop_nums,
            cache_kvs=cache_kvs,
        )
        return outputs

    # rewrite to_static function in generation_utils.py
    def to_static(self, output_path: str, config: dict):
        dtype = config.get("dtype", paddle.get_default_dtype())
        cache_kvs_shapes = self.get_cache_kvs_shape(self.config, max_length=config.get("max_length", None))
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None, None], dtype="float32", name="image_features"
            ),  # image_features
            paddle.static.InputSpec(shape=[None, 3], dtype="int64", name="img_pos"),  # img_pos
            paddle.static.InputSpec(shape=[None, None], dtype=dtype, name="attention_mask"),  # attention_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="position_ids"),  # position_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="penalty_score"),  # penalty_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="frequency_score"),  # frequency_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="presence_score"),  # presence_score
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="min_length"),  # min_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="max_length"),  # max_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="temperature"),  # temperature
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="top_p"),  # top_p
            paddle.static.InputSpec(shape=[None], dtype="int64", name="eos_token_id"),  # eos_token_id
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_encoder"),  # seq_len_encoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_decoder"),  # seq_len_decoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="step_idx"),  # step_idx
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="stop_flags"),  # stop_flags
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_ids"),  # tgt_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_pos"),  # tgt_pos
            paddle.static.InputSpec(
                shape=[None, 1, 1, None], dtype=dtype, name="tgt_generation_mask"
            ),  # tgt_generation_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pre_ids"),  # pre_ids
            paddle.static.InputSpec(shape=[1], dtype="int64", name="stop_nums"),  # stop_nums
            [
                paddle.static.InputSpec(
                    shape=shape,
                    dtype=dtype,
                    name="cache_kvs_{}".format(i),
                )
                for i, shape in enumerate(cache_kvs_shapes)
            ],  # cache_kvs
        ]

        model = paddle.jit.to_static(self.generate_text_with_image_features, input_spec=input_spec)
        paddle.jit.save(model, output_path, skip_prune_program=True)
