# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The EleutherAI Authors and The HuggingFace Inc. team
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
from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddle.nn import Layer

from .. import PretrainedModel, register_base_model
from ..activations import ACT2FN
from ..model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from .configuration import GPTJConfig

__all__ = [
    "GPTJModel",
    "GPTJPretrainedModel",
    "GPTJForCausalLM",
    "GPTJForSequenceClassification",
    "GPTJForQuestionAnswering",
]


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2) / dim))
    sinusoid_inp = paddle.einsum("i , j -> i j", paddle.arange(seq_len, dtype="float32"), inv_freq)
    return paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = paddle.stack((-x2, x1), axis=-1)
    # In einsum notation: rearrange(x, '... d j -> ... (d j)')
    return x.flatten(-2)


def duplicate_interleave(m):
    return paddle.repeat_interleave(m, 2, axis=1)


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(lambda t: duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :], sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class GPTJAttention(Layer):
    def __init__(self, config: GPTJConfig):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            paddle.tril(paddle.ones((max_positions, max_positions), dtype=paddle.get_default_dtype())).reshape(
                (1, 1, max_positions, max_positions)
            ),
        )
        self.register_buffer("masked_bias", paddle.to_tensor(-1e9))
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = paddle.sqrt(paddle.to_tensor(self.head_dim, dtype="float32"))
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.rotary_dim = None
        if config.rotary_dim is not None:
            self.rotary_dim = config.rotary_dim

    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tensor.shape[:-1] + [num_attention_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        if rotary:
            return tensor
        if len(tensor.shape) == 5:

            return tensor.transpose([0, 1, 3, 2, 4])  # (batch, blocks, head, block_length, head_features)
        elif len(tensor.shape) == 4:
            return tensor.transpose([0, 2, 1, 3])  # (batch, head, seq_length, head_features)
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.transpose([0, 1, 3, 2, 4])
        elif len(tensor.shape) == 4:
            tensor = tensor.transpose([0, 2, 1, 3])
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.shape[:-2] + [num_attention_heads * attn_head_size]
        return tensor.reshape(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = query.shape[-2], key.shape[-2]
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = paddle.cast(query, "float32")
        key = paddle.cast(key, "float32")

        attn_weights = paddle.matmul(query, key, transpose_y=True)

        if attn_weights.dtype == paddle.float16:
            mask_value = -65504.0  # smallest representable value for float16
        else:
            mask_value = -1e9  # default value used
        mask_value = paddle.to_tensor(mask_value, dtype=attn_weights.dtype)

        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = paddle.to_tensor(mask_value, dtype=attn_weights.dtype, place=attn_weights.place)
        attn_weights = paddle.where(causal_mask, attn_weights, mask_value)

        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = paddle.nn.functional.softmax(attn_weights, axis=-1)
        attn_weights = attn_weights.astype(value.dtype)

        attn_weights = self.attn_dropout(attn_weights)

        attn_output = paddle.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[paddle.Tensor],
        attention_mask: Optional[paddle.Tensor] = None,
        layer_past: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[paddle.Tensor, Tuple[paddle.Tensor]],
        Optional[Tuple[paddle.Tensor, Tuple[paddle.Tensor], Tuple[paddle.Tensor, ...]]],
    ]:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = paddle.concat([k_rot, k_pass], axis=-1)
            query = paddle.concat([q_rot, q_pass], axis=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.transpose([0, 2, 1, 3])
        query = query.transpose([0, 2, 1, 3])

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = paddle.concat((past_key, key), axis=-2)
            value = paddle.concat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPTJMLP(Layer):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.n_embd

        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, embed_dim)

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[paddle.Tensor]) -> paddle.Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTJBlock(Layer):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, epsilon=config.layer_norm_epsilon)
        self.attn = GPTJAttention(config)
        self.mlp = GPTJMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[paddle.Tensor],
        layer_past: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[paddle.Tensor], Optional[Tuple[paddle.Tensor, Tuple[paddle.Tensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class GPTJPretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTJConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTJBlock"]

    def _init_weights(self, layer):
        """Initialize the weights."""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            if isinstance(layer.weight, paddle.Tensor) and paddle.get_default_dtype() == "float32":
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range")
                        else self.transformer.config["initializer_range"],
                        shape=layer.weight.shape,
                    )
                )
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.full_like(layer.weight, 1.0))
            layer._epsilon = getattr(self, "layer_norm_epsilon", 1e-05)
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            layer.bias.set_value(paddle.zeros_like(layer.bias))


@register_base_model
class GPTJModel(GPTJPretrainedModel):
    def __init__(self, config):
        super(GPTJModel, self).__init__(config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.bos_token_id = config.bos_token_id
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.embed_dim = config.n_embd
        self.initializer_range = config.initializer_range
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.LayerList([GPTJBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_epsilon)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.reshape(shape=(-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape((-1, input_shape[-1]))

        if position_ids is not None:
            position_ids = position_ids.reshape((-1, input_shape[-1]))

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = paddle.arange(past_length, input_shape[-1] + past_length, dtype="int64")
            position_ids = position_ids.unsqueeze(0).reshape((-1, input_shape[-1]))

        # Attention mask.
        if attention_mask is None:
            assert input_ids is not None, "input_ids should be " "specified when generating attention_mask"
            attention_mask = (
                paddle.cast(input_ids == self.pad_token_id, dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e4
            )
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
            attention_mask.stop_gradient = True
        # TODO(zhangxu): Add head_mask if PretrainedModel supports get_head_mask method

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape[:] + [hidden_states.shape[-1]]

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.reshape(shape=output_shape)
        # Add last hidden state
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


class GPTJForCausalLM(GPTJPretrainedModel):
    r"""
    GPTJ Model with a `language modeling` head on top.
    Args:
        GPTJ (:class:`GPTJModel`):
            An instance of GPTJModel.
    """

    def __init__(self, config):
        super(GPTJForCausalLM, self).__init__(config)
        self.transformer = GPTJModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_fast_entry(self, kwargs):
        from paddlenlp.ops import FasterGPTJ

        use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        decoding_lib = kwargs.get("decoding_lib", None)
        decode_strategy = kwargs.get("decode_strategy")
        if decode_strategy == "beam_search":
            raise AttributeError("'beam_search' is not supported yet in the fast version of GPTJ")
        # Currently, FasterTransformer only support restricted size_per_head.
        size_per_head = self.transformer.config["n_embd"] // self.transformer.config["n_head"]
        if size_per_head not in [32, 64, 80, 96, 128, 160, 192, 224, 256]:
            raise AttributeError(
                "'size_per_head = %d' is not supported yet in the fast version of GPTJ" % size_per_head
            )
        if kwargs["forced_bos_token_id"] is not None:
            # not support for min_length yet in the fast version
            raise AttributeError("'forced_bos_token_id != None' is not supported yet in the fast version")
        self._fast_entry = FasterGPTJ(self, decoding_lib=decoding_lib, use_fp16_decoding=use_fp16_decoding).forward
        return self._fast_entry

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask[:, :, -1:, :]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        The GPTJForCausalLM forward method, overrides the __call__() special method.
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import GPTJForCausalLM, GPTJTokenizer
                tokenizer = GPTJTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
                model = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_token_type_ids=False)
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        lm_logits = self.lm_head(hidden_states).astype("float32")

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape([-1, shift_logits.shape[-1]]), shift_labels.reshape([-1]))

            loss = loss.astype(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[paddle.Tensor]], beam_idx: paddle.Tensor) -> Tuple[Tuple[paddle.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.astype(past_state.dtype)) for past_state in layer_past)
            for layer_past in past
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(getattr(self, self.base_model_prefix), name)


class GPTJForSequenceClassification(GPTJPretrainedModel):
    r"""
    GPTJ Model with a linear layer on top of the pooled output,
    designed for sequence classification/regression tasks like GLUE tasks.
    Since it does classification on the last token, it requires to know the
    position of the last token. If a `pad_token_id` is defined in the configuration,
    it finds the last token that is not a padding token in each row. If no `pad_token_id`
    is defined, it simply takes the last value in each row of the batch.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTJModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias_attr=False)

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    paddle.not_equal(
                        input_ids, paddle.to_tensor(self.config.pad_token_id).astype(input_ids.dtype)
                    ).sum(-1)
                    - 1
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[paddle.arange(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels.astype("float32"))

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class GPTJForQuestionAnswering(GPTJPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTJModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        start_positions: Optional[paddle.Tensor] = None,
        end_positions: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = paddle.split(logits, logits.shape[-1], axis=-1)
        start_logits = paddle.squeeze(start_logits, axis=-1)
        end_logits = paddle.squeeze(end_logits, axis=-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
