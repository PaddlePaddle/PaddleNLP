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
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.nn.quant import weight_quantize

from paddlenlp.experimental.transformers.fused_transformer_layers import (
    FusedMultiTransformerConfig,
    FusedMultiTransformerPostLayernorm,
    FusedMultiTransformerWeightOnlyPostLayernorm,
)
from paddlenlp.experimental.transformers.generation_utils import (
    GenerationInferenceModel,
)
from paddlenlp.experimental.transformers.utils import infererence_model_from_pretrained
from paddlenlp.transformers import ChatGLMConfig, ChatGLMPretrainedModel
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithPast,
)
from paddlenlp.transformers.model_utils import (
    dy2st_nocheck_guard_context,
    register_base_model,
)

__all__ = ["ChatGLMForCausalLMInferenceModel"]


def parallel_matmul(lm_output, logit_weights, parallel_output):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()

    if world_size > 1:
        # _c_identity is backwards is reduce
        input_parallel = paddle.distributed.collective._c_identity(lm_output, group=model_parallel_group)

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        # _c_concat has not grad backwards
        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


class RotaryEmbeddingsDybatch(nn.Layer):
    def __init__(self, hidden_size, base=10000.0, learnable=False):
        super().__init__()
        self.dtype = paddle.get_default_dtype()
        inv_freq = 1.0 / (base ** (paddle.arange(0, hidden_size, 2).astype("float32") / hidden_size))
        inv_freq = inv_freq.astype(self.dtype)
        self.learnable = learnable
        if learnable:
            self.inv_freq = nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer("inv_freq", inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None

    def forward(self, seq_dim=1, seq_len=128):
        # TODO: Remove the condition for converting to static graph.
        # if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
        #    self.max_seq_len_cached = None if self.learnable else seq_len

        t = paddle.arange(seq_len).astype(self.dtype)
        # [s, h/n/2]
        # TODO: Failed for fp16 when converting to static graph.
        freqs = paddle.einsum("i,j->ij", t.astype("float32"), self.inv_freq.astype("float32"))

        freqs = freqs.astype(self.dtype)
        # [s, h/n]
        emb = paddle.concat([freqs, freqs], axis=-1)

        if self.dtype == paddle.bfloat16:
            emb = emb.astype("float32")
        # [s, 1, h/n]
        cos_cached = emb.cos().unsqueeze(1)
        sin_cached = emb.sin().unsqueeze(1)

        if self.dtype == paddle.bfloat16:
            cos_cached = cos_cached.astype(self.dtype)
            sin_cached = sin_cached.astype(self.dtype)

        if self.learnable:
            return cos_cached, sin_cached

        self.cos_cached, self.sin_cached = cos_cached, sin_cached

        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


class ChatGLMStackDyBatch(nn.Layer):
    """
    GLM Transformer
    """

    def __init__(self, config: ChatGLMConfig):
        super(ChatGLMStackDyBatch, self).__init__()
        self.config = config
        self.position_encoding_2d = config.position_encoding_2d
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        self.config = config
        self.current_rank = 0
        self.world_size = 1

        self.use_weight_only = False
        if config.quant_type == "weight_only_int8":
            self.use_weight_only = True
            self.quant_algo = "weight_only_int8"
        elif config.quant_type == "weight_only_int4":
            self.use_weight_only = True
            self.quant_algo = "weight_only_int4"

        try:
            self.current_rank = paddle.distributed.get_rank()
            self.world_size = paddle.distributed.get_world_size()
        except Exception:
            pass

        if self.config.tensor_parallel_degree > 1:
            self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.word_embeddings = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        self.rotary_embeddings = RotaryEmbeddingsDybatch(
            self.hidden_size // (self.num_attention_heads * 2)
            if self.position_encoding_2d
            else self.hidden_size // self.num_attention_heads,
            base=10000.0,
        )

        # get ring_id
        ring_id = -1
        try:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            ring_id = model_parallel_group.id
        except:
            pass

        self.input_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layernorm_epsilon)
        ln_scale_attrs = [paddle.ParamAttr(name="fusemt.{}.ln_scale".format(i)) for i in range(config.num_layers)]
        ln_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.ln_bias".format(i)) for i in range(config.num_layers)]
        qkv_weight_attrs = [
            paddle.ParamAttr(
                name="fusemt.{}.qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(config.num_layers)
        ]
        qkv_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.qkv_bias".format(i)) for i in range(config.num_layers)]
        linear_weight_attrs = [
            paddle.ParamAttr(
                name="fusemt.{}.linear_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(config.num_layers)
        ]
        linear_bias_attrs = [
            paddle.ParamAttr(name="fusemt.{}.linear_bias".format(i)) for i in range(config.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="fusemt.{}.ffn_ln_scale".format(i)) for i in range(config.num_layers)
        ]
        ffn_ln_bias_attrs = [
            paddle.ParamAttr(name="fusemt.{}.ffn_ln_bias".format(i)) for i in range(config.num_layers)
        ]
        ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="fusemt.{}.ffn1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(config.num_layers)
        ]
        ffn1_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.ffn1_bias".format(i)) for i in range(config.num_layers)]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fusemt.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(config.num_layers)
        ]
        ffn2_bias_attrs = [paddle.ParamAttr(name="fusemt.{}.ffn2_bias".format(i)) for i in range(config.num_layers)]

        qkv_weight_scale_attrs = None
        linear_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None

        if self.use_weight_only:
            qkv_weight_scale_attrs = [
                paddle.ParamAttr(name="fusemt.{}.qkv_weight_scale".format(i)) for i in range(config.num_layers)
            ]
            linear_weight_scale_attrs = [
                paddle.ParamAttr(name="fusemt.{}.linear_weight_scale".format(i)) for i in range(config.num_layers)
            ]
            ffn1_weight_scale_attrs = [
                paddle.ParamAttr(name="fusemt.{}.ffn1_weight_scale".format(i)) for i in range(config.num_layers)
            ]
            ffn2_weight_scale_attrs = [
                paddle.ParamAttr(name="fusemt.{}.ffn2_weight_scale".format(i)) for i in range(config.num_layers)
            ]

        alpha = (2 * self.config.num_hidden_layers) ** 0.5

        transformer_config = FusedMultiTransformerConfig(
            config.hidden_size,
            config.num_attention_heads,
            4 * config.hidden_size,
            quant_type=config.quant_type,
            activation="gelu",
            num_layers=config.num_layers,
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
            trans_qkvw=True,
            normalize_before=False,
            residual_alpha=alpha,
            norm_type="layernorm",
            use_neox_rotary_style=True,
        )
        if self.use_weight_only:
            self.transformer_block = FusedMultiTransformerWeightOnlyPostLayernorm(transformer_config)
        else:
            self.transformer_block = FusedMultiTransformerPostLayernorm(transformer_config)

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
        pre_caches=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        time_step=None,
        **kwargs,
    ):
        is_decoder = cache is not None
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape[:3]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_lens = seq_len_decoder if is_decoder else seq_len_encoder

        if not is_decoder:
            ids_remove_padding, padding_offset, cum_offsets = self.remove_padding(input_ids, seq_len_encoder)
        else:
            ids_remove_padding = input_ids
            padding_offset = None
            cum_offsets = None

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(ids_remove_padding)

        if cache is None:
            cache = tuple([None] * self.config.num_layers)

        hidden_states = inputs_embeds
        if attention_mask is None:
            attention_mask = paddle.zeros([1, 1]).astype("int64")

        cos, sin = self.rotary_embeddings(seq_len=self.config.max_sequence_length + 1)
        coses = []
        sines = []
        if self.position_encoding_2d:
            block_position_ids = position_ids[:batch_size, 1, :].transpose([1, 0])
            position_ids = position_ids[:batch_size, 0, :].transpose([1, 0])
            coses.append(cos.squeeze(1)[position_ids].unsqueeze(2))
            sines.append(sin.squeeze(1)[position_ids].unsqueeze(2))

            coses.append(cos.squeeze(1)[block_position_ids].unsqueeze(2))
            sines.append(sin.squeeze(1)[block_position_ids].unsqueeze(2))
        else:
            position_ids = position_ids.transpose([1, 0])
            coses.append(cos.squeeze(1)[position_ids].unsqueeze(2))
            sines.append(sin.squeeze(1)[position_ids].unsqueeze(2))

        position_cos = coses[0].transpose([1, 2, 0, 3])
        block_position_cos = coses[1].transpose([1, 2, 0, 3])

        coses = paddle.concat([position_cos, block_position_cos], axis=-1).unsqueeze(0)
        position_sin = sines[0].transpose([1, 2, 0, 3])

        block_position_sin = sines[1].transpose([1, 2, 0, 3])
        sines = paddle.concat([position_sin, block_position_sin], axis=-1).unsqueeze(0)

        rotary_embeds = paddle.concat([coses, sines])

        new_cache = [None]
        hidden_states = self.input_layernorm(hidden_states)

        position_offset = 0
        if not is_decoder and pre_caches is not None:
            position_offset = 128

        with dy2st_nocheck_guard_context():
            hidden_states, new_cache = self.transformer_block(
                input_ids,
                hidden_states,
                cum_offsets=cum_offsets,
                padding_offset=padding_offset,
                attn_mask=paddle.cast(attention_mask, dtype=hidden_states.dtype),
                caches=cache_kvs,
                pre_caches=pre_caches,
                pre_caches_length=position_offset,
                rotary_embs=paddle.cast(rotary_embeds, "float32"),
                rotary_emb_dims=2 if self.config.position_encoding_2d else 1,
                seq_lens=seq_lens,
                time_step=time_step,
            )

        return (hidden_states, new_cache)

    @paddle.no_grad()
    def set_state_dict(self, state_dict, use_structured_name=True):
        self.transformer_block.init_weight()
        dtype = paddle.get_default_dtype()
        config = self.config
        embed_dim = config.hidden_size
        num_attention_heads = config.num_attention_heads // config.tensor_parallel_degree
        head_dim = embed_dim // config.num_attention_heads

        for k, v in state_dict.items():
            if k.startswith("chatglm.transformer.word_embeddings.weight"):
                self.word_embeddings.weight.set_value(v.astype(dtype))
                continue
            elif k.startswith("chatglm.transformer.final_layernorm.weight"):
                self.transformer_block.ffn_ln_scales[config.num_hidden_layers - 1].set_value(v.astype("float32"))
                continue
            elif k.startswith("chatglm.transformer.final_layernorm.bias"):
                self.transformer_block.ffn_ln_biases[config.num_hidden_layers - 1].set_value(v.astype("float32"))
                continue
            elif k.startswith("lm_head.weight"):
                continue
            elif k.endswith("rotary_embeddings.inv_freq") or k.endswith("rotary_emb.inv_freq"):
                continue
            idx = int(k.split(".")[3])
            if k.endswith("input_layernorm.weight"):
                if idx == 0:
                    self.input_layernorm.weight.set_value(v.astype(dtype))
                else:
                    self.transformer_block.ffn_ln_scales[idx - 1].set_value(v.astype("float32"))
            elif k.endswith("input_layernorm.bias"):
                if idx == 0:
                    self.input_layernorm.bias.set_value(v.astype(dtype))
                else:
                    self.transformer_block.ffn_ln_biases[idx - 1].set_value(v.astype("float32"))
            elif k.endswith("post_attention_layernorm.weight"):
                self.transformer_block.ln_scales[idx].set_value(v.astype("float32"))
            elif k.endswith("post_attention_layernorm.bias"):
                self.transformer_block.ln_biases[idx].set_value(v.astype("float32"))
            elif k.endswith("attention.query_key_value.weight"):
                # [embed_dim, num_heads, 3, head_dim] -> [embed_dim, 3, num_heads, head_dim]
                qkv_weight_tensor = (
                    v.reshape([embed_dim, num_attention_heads, 3, head_dim])
                    .transpose([2, 1, 3, 0])
                    .reshape([head_dim * num_attention_heads * 3, embed_dim])
                )

                if self.use_weight_only:
                    qkv_weight_tensor = paddle.transpose(qkv_weight_tensor, perm=[1, 0])
                    qkv_quanted_weight_tensor, qkv_weight_scale_tensor = weight_quantize(
                        qkv_weight_tensor, algo=self.quant_algo
                    )
                    self.transformer_block.qkv_weights[idx].set_value(qkv_quanted_weight_tensor)
                    self.transformer_block.qkv_weights_scale[idx].set_value(qkv_weight_scale_tensor)
                else:
                    self.transformer_block.qkv_weights[idx].set_value(qkv_weight_tensor.astype(dtype))

            elif k.endswith("attention.query_key_value.bias"):
                v = (
                    v.reshape([num_attention_heads, 3, head_dim])
                    .transpose([1, 0, 2])
                    .reshape([head_dim * num_attention_heads * 3])
                )
                self.transformer_block.qkv_biases[idx].set_value(v.astype(dtype))
            elif k.endswith("attention.dense.weight"):
                linear_weight_tensor = v.astype(dtype)
                if self.use_weight_only:
                    linear_quanted_weight_tensor, linear_weight_scale_tensor = weight_quantize(
                        linear_weight_tensor, algo=self.quant_algo
                    )
                    self.transformer_block.linear_weights[idx].set_value(linear_quanted_weight_tensor)
                    self.transformer_block.linear_weights_scale[idx].set_value(linear_weight_scale_tensor)
                else:
                    self.transformer_block.linear_weights[idx].set_value(linear_weight_tensor)

            elif k.endswith("attention.dense.bias"):
                self.transformer_block.linear_biases[idx].set_value(v.astype(dtype))
            elif k.endswith("mlp.dense_h_to_4h.weight"):
                ffn1_weight_tensor = v.astype(dtype)
                if self.use_weight_only:
                    ffn1_quanted_weight_tensor, ffn1_weight_scale_tensor = weight_quantize(
                        ffn1_weight_tensor, algo=self.quant_algo
                    )
                    self.transformer_block.ffn1_weights[idx].set_value(ffn1_quanted_weight_tensor)
                    self.transformer_block.ffn1_weights_scale[idx].set_value(ffn1_weight_scale_tensor)
                else:
                    self.transformer_block.ffn1_weights[idx].set_value(ffn1_weight_tensor)

            elif k.endswith("mlp.dense_h_to_4h.bias"):
                self.transformer_block.ffn1_biases[idx].set_value(v.astype(dtype))
            elif k.endswith("mlp.dense_4h_to_h.weight"):
                ffn2_weight_tensor = v.astype(dtype)
                if self.use_weight_only:
                    ffn2_quanted_weight_tensor, ffn2_weight_scale_tensor = weight_quantize(
                        ffn2_weight_tensor, algo=self.quant_algo
                    )
                    self.transformer_block.ffn2_weights[idx].set_value(ffn2_quanted_weight_tensor)
                    self.transformer_block.ffn2_weights_scale[idx].set_value(ffn2_weight_scale_tensor)
                else:
                    self.transformer_block.ffn2_weights[idx].set_value(ffn2_weight_tensor)

            elif k.endswith("mlp.dense_4h_to_h.bias"):
                self.transformer_block.ffn2_biases[idx].set_value(v.astype(dtype))
            else:
                print("Unknow weight {}".format(k))


@register_base_model
class ChatGLMModelDyBatch(ChatGLMPretrainedModel):
    r"""
    The GLM Model transformer can behave as an encoder (with only self-attention) as well as a decoder, where
    a layer of cross-attention is added between the self-attention layers, following the architecture
    described in [Attention is all you need](https://arxiv.org/abs/1706.03762).

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.
    """

    def __init__(self, config: ChatGLMConfig):
        super(ChatGLMModelDyBatch, self).__init__(config)
        self.config = config
        self.transformer = ChatGLMStackDyBatch(config)
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.transformer.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.transformer.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        cache=None,
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
        time_step=None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = self.get_masks(input_ids)

        if position_ids is None:
            MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id

            use_gmasks = []
            mask_positions = []
            for seq in input_ids:
                mask_token = gMASK if gMASK in seq else MASK
                use_gmask = mask_token == gMASK
                use_gmasks.append(use_gmask)
                mask_positions.append(paddle.where(seq == mask_token)[0][0])
            position_ids = self.get_position_ids(input_ids, mask_positions=mask_positions, use_gmasks=use_gmasks)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        logits, new_caches = self.transformer(
            input_ids=input_ids,
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
            time_step=time_step,
        )

        if not return_dict:
            return (logits, new_caches)

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=logits, past_key_values=new_caches)


class ChatGLMForCausalLMInferenceModel(GenerationInferenceModel, ChatGLMPretrainedModel):
    def __init__(self, config: ChatGLMConfig):
        super(ChatGLMForCausalLMInferenceModel, self).__init__(config)

        self.config = config
        self.max_sequence_length = config.max_sequence_length
        self.position_encoding_2d = config.position_encoding_2d
        self.time_step = paddle.to_tensor([1], dtype="int32", place=paddle.CPUPlace())
        self.model = ChatGLMModelDyBatch(config)

        self.lm_head = self.model.get_input_embeddings()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return infererence_model_from_pretrained(cls, pretrained_model_name_or_path, args, kwargs, return_numpy=False)

    @classmethod
    def get_cache_kvs_shape(
        cls, config: ChatGLMConfig, max_batch_size: int = None, max_length: int = None
    ) -> list[list[int]]:
        """get cache_kvs tensor for llama model

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
        # only last token for inputs_ids if cache is defined in kwargs
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        cache = kwargs.get("cache", None)
        pre_caches = kwargs.get("pre_caches", None)

        time_step = None
        if cache is not None:
            time_step = self.time_step
            input_ids = tgt_ids
            position_ids = tgt_pos
            attention_mask = (1 - tgt_generation_mask) * paddle.finfo(tgt_generation_mask.dtype).min
        else:
            self.time_step = paddle.to_tensor(input_ids.shape[1], dtype="int32", place=paddle.CPUPlace())
            attention_mask = (1 - attention_mask) * paddle.finfo(tgt_generation_mask.dtype).min
            paddle.increment(self.time_step, -1)

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "cache_kvs": cache_kvs,
            "seq_len_encoder": seq_len_encoder,
            "seq_len_decoder": seq_len_decoder,
            "cache": cache,
            "time_step": time_step,
            "pre_caches": pre_caches,
        }
        return model_inputs

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
        pre_caches=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        time_step=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
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
            time_step=time_step,
        )
        hidden_states = transformer_outputs.last_hidden_state if return_dict else transformer_outputs[0]
        if self.config.tensor_parallel_degree > 1:
            lm_logits = parallel_matmul(hidden_states, self.lm_head.weight, self.config.tensor_parallel_output)
        else:
            lm_logits = F.linear(hidden_states, self.lm_head.weight.T)

        loss = None
        if labels is not None:
            """
            for p, l in zip(lm_logits[..., :-1, :].argmax(axis=-1), labels[..., 1:]):
                print("prediction")
                print(self.tokenizer.decode(p[l != -100].tolist()))
                print("labels")
                print(self.tokenizer.decode(l[l != -100].tolist()))
            """

            shift_logits = lm_logits[..., :-1, :]
            shift_logits = shift_logits.reshape([-1, shift_logits.shape[-1]])
            shift_logits = shift_logits.astype("float32")
            shift_labels = labels[..., 1:].reshape([-1])

            if self.config.tensor_parallel_degree > 1 and self.config.tensor_parallel_output:
                self.parallel_loss_func = fleet.meta_parallel.ParallelCrossEntropy()
                shift_logits = shift_logits[shift_labels != -100]
                shift_labels = shift_labels[shift_labels != -100]
                loss = self.parallel_loss_func(shift_logits, shift_labels).mean()
            else:
                loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            loss = loss.astype(lm_logits.dtype)
        if time_step:
            paddle.increment(self.time_step)

        if not return_dict:
            if loss is not None:
                return (loss, lm_logits, transformer_outputs[1:])
            else:
                return (lm_logits, transformer_outputs[1:])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.lm_head.weight.set_value(
            state_dict["chatglm.transformer.word_embeddings.weight"].astype(self.lm_head.weight.dtype)
        )
        self.model.transformer.set_state_dict({k: state_dict[k] for k in state_dict.keys()})
