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

from typing import Tuple, Union

import paddle
from paddle import Tensor, nn
from paddle.distributed import fleet
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
from paddlenlp.transformers.bloom.modeling import BloomPreTrainedModel
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import (
    dy2st_nocheck_guard_context,
    register_base_model,
)

__all__ = [
    "BloomModelInferenceModel",
    "BloomForCausalLMInferenceModel",
    "BloomBlockInferenceModel",
    "BloomForCausalLMBlockInferenceModel",
]


def parallel_matmul(x: Tensor, y: Tensor, parallel_output=True):
    is_fleet_init = True
    world_size = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        world_size = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False
    if is_fleet_init and world_size > 1:
        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        input_parallel = paddle.distributed.collective._c_identity(x, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, y, transpose_y=True)
        if parallel_output:
            return logits
        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(x, y, transpose_y=True)
        return logits


@register_base_model
class BloomModelInferenceModel(BloomPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = 0

        self.embed_dim = config.hidden_size
        self.n_head = config.n_head

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
            ), "Expected quant_algo equal to 'weight_only_int8' or 'weight_only_int4', but received {}".format(
                self.quant_algo
            )

        # Embedding + LN Embedding
        if config.tensor_parallel_degree > 1:
            self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Normal(mean=0.0, std=config.initializer_range)
                ),
            )
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        self.word_embeddings_layernorm = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_epsilon)

        # get ring_id
        ring_id = -1
        try:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            ring_id = model_parallel_group.id
        except:
            pass

        # Transformer blocks
        ln_scale_attrs = [paddle.ParamAttr(name="fusemt.{}.ln_scale".format(i)) for i in range(config.n_layer)]
        ln_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.ln_bias".format(i)) for i in range(config.n_layer)]
        qkv_weight_attrs = [
            paddle.ParamAttr(
                name="fusemt.{}.qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(config.n_layer)
        ]
        qkv_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.qkv_bias".format(i)) for i in range(config.n_layer)]
        linear_weight_attrs = [
            paddle.ParamAttr(
                name="fusemt.{}.linear_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(config.n_layer)
        ]
        linear_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.linear_bias".format(i)) for i in range(config.n_layer)]
        ffn_ln_scale_attrs = [paddle.ParamAttr(name="fusemt.{}.ffn_ln_scale".format(i)) for i in range(config.n_layer)]
        ffn_ln_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.ffn_ln_bias".format(i)) for i in range(config.n_layer)]
        ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="fusemt.{}.ffn1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(config.n_layer)
        ]
        ffn1_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.ffn1_bias".format(i)) for i in range(config.n_layer)]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fusemt.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(config.n_layer)
        ]
        ffn2_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.ffn2_bias".format(i)) for i in range(config.n_layer)]
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
            self.embed_dim,
            self.n_head,
            4 * self.embed_dim,
            quant_type=config.quant_type,
            activation="gelu",
            num_layers=config.n_layer,
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
        )

        self.set_transformer_block(transformer_config)

        self.cache_kvs = []

        # Final Layer Norm
        self.ln_f = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

    def set_transformer_block(self, transformer_config):
        if self.use_weight_only:
            self.transformer_block = FusedMultiTransformerWeightOnly(transformer_config)
        else:
            self.transformer_block = FusedMultiTransformerBase(transformer_config)

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: Tensor):
        self.word_embeddings = new_embeddings

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
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        cache=None,
        cache_kvs=None,
        pre_caches=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        return_dict=None,
        **kwargs,
    ) -> Union[Tuple[Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # past_key_values = kwargs.get("cache", past_key_values)
        # is_decoder = past_key_values is not None
        is_decoder = cache is not None
        seq_len = seq_len_decoder if is_decoder else seq_len_encoder
        if not is_decoder:
            ids_remove_padding, padding_offset, cum_offsets = self.remove_padding(input_ids, seq_len)
        else:
            ids_remove_padding = input_ids
            padding_offset = None
            cum_offsets = None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(ids_remove_padding)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        position_offset = 0
        if not is_decoder and pre_caches is not None:
            position_offset = 128

        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                src=hidden_states,
                input_ids=input_ids,
                cum_offsets=cum_offsets,
                padding_offset=padding_offset,
                attn_mask=paddle.cast(attention_mask, dtype=hidden_states.dtype),
                caches=cache_kvs,
                pre_caches=pre_caches,
                pre_caches_length=position_offset,
                seq_lens=seq_len,
                time_step=paddle.increment(paddle.shape(attention_mask)[-1], -1) if is_decoder else None,
            )

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states)

    @paddle.no_grad()
    def set_state_dict(self, state_dict, use_structured_name=True):
        self.transformer_block.init_weight()
        for k, v in state_dict.items():
            if k.find("word_embeddings.weight") >= 0:
                self.word_embeddings.weight.set_value(paddle.to_tensor(v))
            elif k.find("word_embeddings_layernorm.weight") >= 0:
                self.word_embeddings_layernorm.weight.set_value(paddle.to_tensor(v))
            elif k.find("word_embeddings_layernorm.bias") >= 0:
                self.word_embeddings_layernorm.bias.set_value(paddle.to_tensor(v))
            elif k.find("ln_f.weight") >= 0:
                self.ln_f.weight.set_value(paddle.to_tensor(v))
            elif k.find("ln_f.bias") >= 0:
                self.ln_f.bias.set_value(paddle.to_tensor(v))
            else:
                # transformer block weights
                splits = k.split(".")
                idx = int(splits[1]) if splits[1].isdigit() else int(splits[2])

                if k.endswith("input_layernorm.weight"):
                    self.transformer_block.ln_scales[idx].set_value(paddle.to_tensor(v).astype("float32"))
                elif k.endswith("input_layernorm.bias"):
                    self.transformer_block.ln_biases[idx].set_value(paddle.to_tensor(v).astype("float32"))
                elif k.endswith("self_attention.query_key_value.weight"):
                    qkv_weight_tensor = (
                        v.reshape(
                            [
                                self.embed_dim,
                                self.n_head // self.config.tensor_parallel_degree,
                                3,
                                self.embed_dim // self.n_head,
                            ]
                        )
                        .transpose([2, 1, 3, 0])
                        .reshape([-1, self.embed_dim])
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
                elif k.endswith("self_attention.query_key_value.bias"):
                    v = (
                        v.reshape(
                            [
                                self.n_head // self.config.tensor_parallel_degree,
                                3,
                                self.embed_dim // self.n_head,
                            ]
                        )
                        .transpose([1, 0, 2])
                        .reshape([-1])
                    )
                    self.transformer_block.qkv_biases[idx].set_value(paddle.to_tensor(v))
                elif k.endswith("self_attention.dense.weight"):
                    linear_weight_tensor = paddle.to_tensor(v)
                    if self.use_weight_only:
                        linear_quanted_weight_tensor, linear_weight_scale_tensor = weight_quantize(
                            linear_weight_tensor, algo=self.quant_algo
                        )
                        self.transformer_block.linear_weights[idx].set_value(linear_quanted_weight_tensor)
                        self.transformer_block.linear_weights_scale[idx].set_value(linear_weight_scale_tensor)
                    else:
                        self.transformer_block.linear_weights[idx].set_value(linear_weight_tensor)
                elif k.endswith("self_attention.dense.bias"):
                    self.transformer_block.linear_biases[idx].set_value(paddle.to_tensor(v))
                elif k.endswith("post_attention_layernorm.weight"):
                    self.transformer_block.ffn_ln_scales[idx].set_value(paddle.to_tensor(v).astype("float32"))
                elif k.endswith("post_attention_layernorm.bias"):
                    self.transformer_block.ffn_ln_biases[idx].set_value(paddle.to_tensor(v).astype("float32"))
                elif k.endswith("mlp.dense_h_to_4h.weight"):
                    ffn1_weight_tensor = paddle.to_tensor(v)
                    if self.use_weight_only:
                        ffn1_quanted_weight_tensor, ffn1_weight_scale_tensor = weight_quantize(
                            ffn1_weight_tensor, algo=self.quant_algo
                        )
                        self.transformer_block.ffn1_weights[idx].set_value(ffn1_quanted_weight_tensor)
                        self.transformer_block.ffn1_weights_scale[idx].set_value(ffn1_weight_scale_tensor)
                    else:
                        self.transformer_block.ffn1_weights[idx].set_value(ffn1_weight_tensor)
                elif k.endswith("mlp.dense_h_to_4h.bias"):
                    self.transformer_block.ffn1_biases[idx].set_value(paddle.to_tensor(v))
                elif k.endswith("mlp.dense_4h_to_h.weight"):
                    ffn2_weight_tensor = paddle.to_tensor(v)
                    if self.use_weight_only:
                        ffn2_quanted_weight_tensor, ffn2_weight_scale_tensor = weight_quantize(
                            ffn2_weight_tensor, algo=self.quant_algo
                        )
                        self.transformer_block.ffn2_weights[idx].set_value(ffn2_quanted_weight_tensor)
                        self.transformer_block.ffn2_weights_scale[idx].set_value(ffn2_weight_scale_tensor)
                    else:
                        self.transformer_block.ffn2_weights[idx].set_value(ffn2_weight_tensor)

                elif k.endswith("mlp.dense_4h_to_h.bias"):
                    self.transformer_block.ffn2_biases[idx].set_value(paddle.to_tensor(v))
                else:
                    raise ValueError("Unknow weight {}".format(k))


class BloomLMHead(nn.Layer):
    def __init__(self, config, embedding_weights=None):
        super(BloomLMHead, self).__init__()
        self.decoder_weight = (
            self.create_parameter(
                shape=[config.vocab_size, config.hidden_size],
                dtype=paddle.get_default_dtype(),
                is_bias=True,
            )
            if embedding_weights is None
            else embedding_weights
        )
        self.config = config

    def forward(self, hidden_states):
        logits = parallel_matmul(hidden_states, self.decoder_weight, parallel_output=False)
        return logits


class BloomPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for GPT.
    It calculates the final loss.
    """

    def __init__(self, pad_token_id=None, tensor_parallel_degree=1, tensor_parallel_output=False):
        super(BloomPretrainingCriterion, self).__init__()
        if tensor_parallel_degree > 1 and tensor_parallel_output:
            self.loss_func = fleet.meta_parallel.ParallelCrossEntropy()
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
        self.pad_token_id = pad_token_id

    def forward(self, prediction_scores, masked_lm_labels, loss_mask=None):
        masked_lm_loss = self.loss_func(prediction_scores, masked_lm_labels.unsqueeze(2))
        with paddle.amp.auto_cast(False):
            masked_lm_loss = masked_lm_loss.astype("float32")
            if loss_mask is not None:
                loss_mask = loss_mask.reshape([-1])
                masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
                loss = masked_lm_loss / loss_mask.sum()
            else:
                assert self.pad_token_id is not None
                masked_lm_loss = masked_lm_loss[masked_lm_labels != self.pad_token_id]
                loss = paddle.mean(masked_lm_loss)

        return loss


class BloomForCausalLMInferenceModel(GenerationInferenceModel, BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h.*.self_attention.scale_mask_softmax.causal_mask",
        r"lm_head.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.bloom = BloomModelInferenceModel(config)
        self.lm_head = BloomLMHead(config, self.bloom.word_embeddings.weight)
        self.criterion = BloomPretrainingCriterion(
            pad_token_id=config.pad_token_id,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_output=True,
        )

    @classmethod
    def get_cache_kvs_shape(cls, config, max_batch_size=None, max_length=None) -> list[list[int]]:
        """get cache_kvs tensor for llama model

        Args:
            max_batch_size (int): the max batch size
            max_length (int | None, optional): the max_length of cache_kvs. Defaults to None.

        Returns:
            list[paddle.Tensor]: the list tensor shape for cache
        """
        if max_length is None:
            max_length = 2048

        cache_kvs = []
        for _ in range(config.n_layer):
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

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, cache_kvs, tgt_ids, tgt_generation_mask, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        pre_caches = kwargs.get("pre_caches", None)
        seq_len_encoder = kwargs.get("seq_len_encoder", None)
        seq_len_decoder = kwargs.get("seq_len_decoder", None)
        cache = kwargs.get("cache", None)
        if cache is not None:
            input_ids = tgt_ids
            attention_mask = tgt_generation_mask
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_kvs": cache_kvs,
            "cache": cache,
            "pre_caches": pre_caches,
            "use_cache": True,
            "seq_len_encoder": seq_len_encoder,
            "seq_len_decoder": seq_len_decoder,
        }

    def forward(
        self,
        input_ids=None,
        cache=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_kvs=None,
        pre_caches=None,
        output_attentions=None,
        output_hidden_states=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        return_dict=None,
    ) -> Union[Tuple[Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.bloom(
            input_ids,
            cache=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_kvs=cache_kvs,
            pre_caches=pre_caches,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return output

        return CausalLMOutputWithCrossAttentions(logits=lm_logits)

    @paddle.no_grad()
    def set_state_dict(self, state_dict, use_structured_name=True):
        self.lm_head.set_state_dict(
            {k: state_dict[k] for k in state_dict.keys() if "lm_head" in k},
            use_structured_name,
        )
        self.bloom.set_state_dict({k: state_dict[k] for k in state_dict.keys() if "bloom" in k})

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[Tensor]], beam_idx: Tensor) -> Tuple[Tuple[Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(tuple(past_state.index_select(0, beam_idx) for past_state in layer_past) for layer_past in past)


@register_base_model
class BloomBlockInferenceModel(BloomModelInferenceModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_len = config.max_seq_len
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
        ids_remove_padding, padding_offset, cum_offsets, cu_seqlens_q, cu_seqlens_k = self.remove_padding(
            input_ids, seq_lens_this_time
        )
        kwargs["cu_seqlens_q"] = cu_seqlens_q
        kwargs["cu_seqlens_k"] = cu_seqlens_k
        kwargs["padding_offsets"] = padding_offset
        kwargs["max_input_length"] = self.max_seq_len

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(ids_remove_padding)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                input_ids=input_ids,
                src=hidden_states,
                cum_offsets=cum_offsets,
                attn_mask=attention_mask,
                caches=caches,
                pre_caches=pre_caches,
                rotary_embs=None,
                **kwargs,
            )

        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class BloomForCausalLMBlockInferenceModel(GenerationBlockInferenceModel, BloomPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bloom = BloomBlockInferenceModel(config)
        self.lm_head = BloomLMHead(config, self.bloom.word_embeddings.weight)

    @classmethod
    def get_cache_kvs_shape(cls, config, max_batch_size: int = None, max_length: int = None):

        max_block_per_seq = (config.max_seq_len + config.block_size - 1) // config.block_size
        if max_batch_size == -1:
            max_block_nums = None
        else:
            max_block_nums = max_batch_size * max_block_per_seq

        cache_kvs = []
        for _ in range(config.n_layer):
            cache_kv_shape = [
                max_block_nums,
                config.n_head // max(config.tensor_parallel_degree, 1),
                config.block_size,
                config.hidden_size // config.n_head,
            ]
            cache_kvs.append(cache_kv_shape)
            cache_kvs.append(cache_kv_shape)
        return cache_kvs

    def prepare_inputs_for_generation(self, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        input_ids = kwargs["input_ids"]
        src_mask = kwargs.get("src_mask", None)

        tgt_mask = kwargs.get("tgt_mask", None)

        block_tables = kwargs.get("block_tables", None)

        pre_caches = kwargs.get("pre_caches", None)
        caches = kwargs.get("caches", None)

        seq_lens_this_time = kwargs["seq_lens_this_time"]

        seq_lens_encoder = kwargs["seq_lens_encoder"]
        seq_lens_decoder = kwargs["seq_lens_decoder"]
        k_quant_scales = kwargs.get("k_quant_scales", None)
        v_quant_scales = kwargs.get("v_quant_scales", None)
        k_dequant_scales = kwargs.get("k_dequant_scales", None)
        v_dequant_scales = kwargs.get("v_dequant_scales", None)

        # only slice a part of src_mask, because of phi::FlashAttnUnpaddedKernel.
        valid_max_encoder_len = paddle.max(seq_lens_encoder)
        src_mask = src_mask[:, :, :valid_max_encoder_len, :valid_max_encoder_len]

        model_inputs = {
            "input_ids": input_ids,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "rope_emb": None,
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
        tgt_mask=None,
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
        outputs = self.bloom(
            input_ids,
            attention_mask=src_mask,
            tgt_mask=tgt_mask,
            caches=caches,
            # bloom does not have rope_emb!
            rope_emb=None,
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

        output = self.lm_head(hidden_states)

        return output

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.bloom.set_state_dict(state_dict)
