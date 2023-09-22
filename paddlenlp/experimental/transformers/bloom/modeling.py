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
from paddlenlp_ops import get_padding_offset

from paddlenlp.experimental.transformers.fused_transformer_layers import (
    FusedMultiTransformer,
)
from paddlenlp.experimental.transformers.generation_utils import (
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
        qkv_weight_attrs = [paddle.ParamAttr(name="fusemt.{}.qkv_weight".format(i)) for i in range(config.n_layer)]
        qkv_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.qkv_bias".format(i)) for i in range(config.n_layer)]
        linear_weight_attrs = [
            paddle.ParamAttr(name="fusemt.{}.linear_weight".format(i)) for i in range(config.n_layer)
        ]
        linear_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.linear_bias".format(i)) for i in range(config.n_layer)]
        ffn_ln_scale_attrs = [paddle.ParamAttr(name="fusemt.{}.ffn_ln_scale".format(i)) for i in range(config.n_layer)]
        ffn_ln_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.ffn_ln_bias".format(i)) for i in range(config.n_layer)]
        ffn1_weight_attrs = [paddle.ParamAttr(name="fusemt.{}.ffn1_weight".format(i)) for i in range(config.n_layer)]
        ffn1_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.ffn1_bias".format(i)) for i in range(config.n_layer)]
        ffn2_weight_attrs = [paddle.ParamAttr(name="fusemt.{}.ffn2_weight".format(i)) for i in range(config.n_layer)]
        ffn2_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.ffn2_bias".format(i)) for i in range(config.n_layer)]
        self.transformer_block = FusedMultiTransformer(
            self.embed_dim,
            self.n_head,
            4 * self.embed_dim,
            activation="gelu",
            num_layers=config.n_layer,
            nranks=config.tensor_parallel_degree,
            ring_id=ring_id,
            ln_scale_attrs=ln_scale_attrs,
            ln_bias_attrs=ln_bias_attrs,
            qkv_weight_attrs=qkv_weight_attrs,
            qkv_bias_attrs=qkv_bias_attrs,
            linear_weight_attrs=linear_weight_attrs,
            linear_bias_attrs=linear_bias_attrs,
            ffn_ln_scale_attrs=ffn_ln_scale_attrs,
            ffn_ln_bias_attrs=ffn_ln_bias_attrs,
            ffn1_weight_attrs=ffn1_weight_attrs,
            ffn1_bias_attrs=ffn1_bias_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_bias_attrs=ffn2_bias_attrs,
        )
        self.cache_kvs = []

        # Final Layer Norm
        self.ln_f = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: Tensor):
        self.word_embeddings = new_embeddings

    def remove_padding(self, input_ids, seq_lens_this_time):
        cum_offsets_now = paddle.cumsum(paddle.max(seq_lens_this_time) - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
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

        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                src=hidden_states,
                input_ids=input_ids,
                cum_offsets=cum_offsets,
                padding_offset=padding_offset,
                attn_mask=paddle.cast(attention_mask, dtype=hidden_states.dtype),
                caches=cache_kvs,
                seq_lens=seq_len,
                time_step=paddle.increment(paddle.shape(attention_mask)[-1], -1) if is_decoder else None,
            )

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states)

    @paddle.no_grad()
    def set_state_dict(self, state_dict, use_structured_name=True):

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
                    v = (
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
                    self.transformer_block.qkv_weights[idx].set_value(paddle.to_tensor(v))
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
                    self.transformer_block.linear_weights[idx].set_value(paddle.to_tensor(v))
                elif k.endswith("self_attention.dense.bias"):
                    self.transformer_block.linear_biases[idx].set_value(paddle.to_tensor(v))
                elif k.endswith("post_attention_layernorm.weight"):
                    self.transformer_block.ffn_ln_scales[idx].set_value(paddle.to_tensor(v).astype("float32"))
                elif k.endswith("post_attention_layernorm.bias"):
                    self.transformer_block.ffn_ln_biases[idx].set_value(paddle.to_tensor(v).astype("float32"))
                elif k.endswith("mlp.dense_h_to_4h.weight"):
                    self.transformer_block.ffn1_weights[idx].set_value(paddle.to_tensor(v))
                elif k.endswith("mlp.dense_h_to_4h.bias"):
                    self.transformer_block.ffn1_biases[idx].set_value(paddle.to_tensor(v))
                elif k.endswith("mlp.dense_4h_to_h.weight"):
                    self.transformer_block.ffn2_weights[idx].set_value(paddle.to_tensor(v))
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
