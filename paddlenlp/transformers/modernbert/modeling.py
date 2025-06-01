# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import math
from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer

from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model
from paddlenlp.utils.converter import StateDictNameMapping, init_name_mappings
from paddlenlp.utils.log import logger

from ..model_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from .configuration import (
    MODERNBERT_PRETRAINED_INIT_CONFIGURATION,
    MODERNBERT_PRETRAINED_RESOURCE_FILES_MAP,
    ModernBertConfig,
)

__all__ = [
    "ModernBertModel",
    "ModernBertPretrainedModel",
    "ModernBertForSequenceClassification",
    "ModernBertForTokenClassification",
    "ModernBertForQuestionAnswering",
    "ModernBertForMultipleChoice",
    "ModernBertForMaskedLM",
]


def geglu(x: Tensor, gate: Tensor) -> Tensor:
    """Gated GLU activation function.
    
    GeGLU(x, gate) = x * GELU(gate)
    
    Args:
        x: Input tensor
        gate: Gate tensor
        
    Returns:
        Tensor: Output after applying GeGLU activation
    """
    return x * F.gelu(gate)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key tensors.
    
    Implements RoPE (Rotary Position Embedding) as described in the paper:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    Args:
        q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
        cos: Cosine of position embeddings
        sin: Sine of position embeddings 
        position_ids: Position IDs tensor
        
    Returns:
        Tuple[Tensor, Tensor]: Rotated query and key tensors
    """
    # Reshape q and k to match the complex representation
    q_real, q_imag = q.chunk(2, axis=-1)
    k_real, k_imag = k.chunk(2, axis=-1)
    
    # Gather cos and sin values according to position_ids
    cos = paddle.gather(cos, position_ids.flatten(), axis=0).reshape(position_ids.shape + [cos.shape[-1]])
    sin = paddle.gather(sin, position_ids.flatten(), axis=0).reshape(position_ids.shape + [sin.shape[-1]])
    cos = cos.unsqueeze(2)  # [bs, seq_len, 1, dim]
    sin = sin.unsqueeze(2)  # [bs, seq_len, 1, dim]
    
    # Apply rotary position embeddings
    q_out_real = q_real * cos - q_imag * sin
    q_out_imag = q_real * sin + q_imag * cos
    k_out_real = k_real * cos - k_imag * sin
    k_out_imag = k_real * sin + k_imag * cos
    
    # Concatenate real and imaginary parts
    q_out = paddle.concat([q_out_real, q_out_imag], axis=-1)
    k_out = paddle.concat([k_out_real, k_out_imag], axis=-1)
    
    return q_out, k_out


class ModernBertSelfAttention(Layer):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.rotary_value = getattr(config, "rotary_value", False)
        self.max_position_embeddings = config.max_position_embeddings
        self.sliding_window_size = config.sliding_window_size
        
        # Initialize rotary position embeddings
        if self.position_embedding_type == "rotary":
            self.rotary_dim = self.attention_head_size // 2
            self.register_buffer(
                "cos_cached",
                paddle.cos(self._get_rotary_positions(self.max_position_embeddings)),
                persistable=False,
            )
            self.register_buffer(
                "sin_cached",
                paddle.sin(self._get_rotary_positions(self.max_position_embeddings)),
                persistable=False,
            )
    
    def _get_rotary_positions(self, seq_length: int) -> Tensor:
        """Generate position indices for rotary position embeddings."""
        position = paddle.arange(0, seq_length, dtype=paddle.float32)
        dim = paddle.arange(0, self.rotary_dim, 2, dtype=paddle.float32)
        dim = 1.0 / paddle.pow(10000, dim / self.rotary_dim)
        position = paddle.einsum("i,j->ij", position, dim)
        return position
    
    def _sliding_chunks_query_key_matmul(
        self, query: Tensor, key: Tensor, window_overlap: int
    ) -> Tensor:
        """Matrix multiplication of query and key tensors using sliding window attention.
        
        This implementation splits the input into overlapping chunks of size 2w (window_overlap)
        with an overlap of size w."""
        batch_size, seq_length, num_heads, head_dim = query.shape
        chunks_count = seq_length // window_overlap - 1

        # Group batch_size and num_heads dimensions into one
        query = query.reshape([batch_size * num_heads, seq_length, head_dim])
        key = key.reshape([batch_size * num_heads, seq_length, head_dim])

        # Compute attention scores for each window
        chunks_q = paddle.stack(
            [query[:, i:i + window_overlap * 2] for i in range(0, seq_length - window_overlap, window_overlap)],
            axis=1
        )
        chunks_k = paddle.stack(
            [key[:, i:i + window_overlap * 2] for i in range(0, seq_length - window_overlap, window_overlap)],
            axis=1
        )

        # Matrix multiplication
        attention_scores = paddle.matmul(chunks_q, chunks_k.transpose([0, 1, 3, 2]))
        
        # Reshape attention scores to original dimensions
        attention_scores = attention_scores.reshape([batch_size, num_heads, chunks_count, window_overlap * 2, window_overlap * 2])
        
        return attention_scores

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: Optional[bool] = False,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = paddle.concat([past_key_value[0], key_layer], axis=2)
            value_layer = paddle.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = apply_rotary_pos_emb(
                query_layer,
                key_layer,
                self.cos_cached,
                self.sin_cached,
                position_ids,
            )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        if self.sliding_window_size is not None:
            attention_scores = self._sliding_chunks_query_key_matmul(
                query_layer, key_layer, self.sliding_window_size
            )
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # Apply the attention mask
            if attention_mask is not None:
                # Apply sliding window attention mask
                window_mask = self._create_sliding_window_mask(
                    attention_mask, self.sliding_window_size
                )
                attention_scores = attention_scores + window_mask
        else:
            attention_scores = paddle.matmul(query_layer, key_layer.transpose([0, 1, 3, 2]))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def _create_sliding_window_mask(self, attention_mask: Tensor, window_size: int) -> Tensor:
        """Create a mask for sliding window attention.
        
        Args:
            attention_mask: Original attention mask of shape [batch_size, 1, seq_len, seq_len]
            window_size: Size of the sliding window
            
        Returns:
            Tensor: Modified attention mask that implements sliding window attention
        """
        batch_size, _, seq_len, _ = attention_mask.shape
        
        # Create a mask that only allows attention within the sliding window
        window_mask = paddle.ones([seq_len, seq_len], dtype=attention_mask.dtype) * float("-inf")
        for i in range(seq_len):
            window_start = max(0, i - window_size)
            window_end = min(seq_len, i + window_size + 1)
            window_mask[i, window_start:window_end] = 0
        
        # Expand mask to match attention_mask shape
        window_mask = window_mask.expand([batch_size, 1, seq_len, seq_len])
        
        return window_mask

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])
            
            # Compute rotary embeddings
            sinusoid_inp = paddle.einsum('i,j->ij', seq_idx, theta)
            self.register_buffer("cos_cached", paddle.cos(sinusoid_inp))
            self.register_buffer("sin_cached", paddle.sin(sinusoid_inp))

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: Optional[bool] = False,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = paddle.concat([past_key_value[0], key_layer], axis=2)
            value_layer = paddle.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = apply_rotary_pos_emb(
                query_layer, key_layer, self.cos_cached, self.sin_cached, position_ids
            )
            if self.rotary_value:
                _, value_layer = apply_rotary_pos_emb(
                    value_layer, value_layer, self.cos_cached, self.sin_cached, position_ids
                )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_layer, key_layer.transpose([0, 1, 3, 2]))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.shape[1]
            position_ids_l = paddle.arange(seq_length, dtype="int64").reshape([-1, 1])
            position_ids_r = paddle.arange(seq_length, dtype="int64").reshape([1, -1])
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.transpose([2, 0, 1]).expand([attention_scores.shape[0], -1, -1, -1])

            if self.position_embedding_type == "relative_key":
                relative_position_scores = paddle.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = paddle.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = paddle.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if self.window_size is not None:
            # Apply sliding window mask
            batch_size, num_heads, seq_len, _ = attention_scores.shape
            mask = paddle.ones([seq_len, seq_len], dtype=attention_scores.dtype)
            mask = paddle.triu(mask, diagonal=-self.window_size) * paddle.tril(mask, diagonal=self.window_size)
            mask = mask.unsqueeze([0, 1]).expand([batch_size, num_heads, -1, -1])
            attention_scores = paddle.where(mask == 1, attention_scores, paddle.full_like(attention_scores, float("-inf")))

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if past_key_value is not None:
            outputs = outputs + (key_layer, value_layer)

        return outputs


class ModernBertSelfOutput(Layer):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ModernBertAttention(Layer):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.self = ModernBertSelfAttention(config)
        self.output = ModernBertSelfOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: Optional[bool] = False,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            position_ids,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ModernBertIntermediate(Layer):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = getattr(F, config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ModernBertOutput(Layer):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ModernBertLayer(Layer):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ModernBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ModernBertAttention(config)
        self.intermediate = ModernBertIntermediate(config)
        self.output = ModernBertOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[Tensor]]] = None,
        output_attentions: Optional[bool] = False,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            position_ids=position_ids,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs


class ModernBertEncoder(Layer):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.layer = nn.LayerList([ModernBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        position_ids: Optional[Tensor] = None,
    ) -> Union[Tuple[Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                position_ids,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class ModernBertPooler(Layer):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@register_base_model
class ModernBertModel(ModernBertPretrainedModel):
    """The bare ModernBERT Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/fluid/dygraph/layers/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`ModernBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~paddlenlp.transformers.PretrainedModel.from_pretrained` method to load the model weights.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = ModernBertEmbeddings(config)
        self.encoder = ModernBertEncoder(config)
        self.pooler = ModernBertPooler(config)

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel"""
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = paddle.ones(((batch_size, seq_length + past_key_values_length)))

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand((batch_size, seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = paddle.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_ids=position_ids,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class ModernBertEmbeddings(Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.padding_idx = config.pad_token_id
        self.position_embeddings_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_values_length: int = 0,
    ) -> Tensor:
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = paddle.arange(past_key_values_length, seq_length + past_key_values_length, dtype="int64")
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embeddings_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ModernBertPretrainedModel(PretrainedModel):
    config_class = ModernBertConfig
    base_model_prefix = "modernbert"
    pretrained_init_configuration = MODERNBERT_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = MODERNBERT_PRETRAINED_RESOURCE_FILES_MAP
    _keys_to_ignore_on_load_unexpected = [r"pooler\.", r"cls\."]

    def _init_weights(self, layer):
        """Initialize the weights."""
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.tensor.normal(mean=0.0, std=self.config.initializer_range, shape=layer.weight.shape)
            )
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(
                paddle.tensor.normal(mean=0.0, std=self.config.initializer_range, shape=layer.weight.shape)
            )
            if layer._padding_idx is not None:
                layer.weight[layer._padding_idx].set_value(paddle.zeros_like(layer.weight[layer._padding_idx]))
        elif isinstance(layer, nn.LayerNorm):
            layer.weight.set_value(paddle.ones_like(layer.weight))
            layer.bias.set_value(paddle.zeros_like(layer.bias))

    @classmethod
    def _get_name_mappings(cls, config: ModernBertConfig) -> list[StateDictNameMapping]:
        """Get name mappings for loading HuggingFace weights into PaddleNLP.
        
        Args:
            config: Model configuration.
            
        Returns:
            List of StateDictNameMapping objects for parameter conversion.
        """
        mappings = [
            "embeddings.LayerNorm.weight",
            "embeddings.LayerNorm.bias",
            "embeddings.position_ids",
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
            "pooler.dense.weight",
            "pooler.dense.bias",
            "cls.predictions.transform.dense.weight",
            "cls.predictions.transform.dense.bias",
            "cls.predictions.transform.LayerNorm.weight",
            "cls.predictions.transform.LayerNorm.bias",
            "cls.predictions.decoder.weight",
            "cls.predictions.decoder.bias",
            "cls.predictions.bias",
        ]
        
        # Add encoder layer mappings
        for i in range(config.num_hidden_layers):
            layer_mappings = [
                f"encoder.layer.{i}.attention.self.query.weight",
                f"encoder.layer.{i}.attention.self.query.bias",
                f"encoder.layer.{i}.attention.self.key.weight",
                f"encoder.layer.{i}.attention.self.key.bias",
                f"encoder.layer.{i}.attention.self.value.weight",
                f"encoder.layer.{i}.attention.self.value.bias",
                f"encoder.layer.{i}.attention.output.dense.weight",
                f"encoder.layer.{i}.attention.output.dense.bias",
                f"encoder.layer.{i}.attention.output.LayerNorm.weight",
                f"encoder.layer.{i}.attention.output.LayerNorm.bias",
                f"encoder.layer.{i}.intermediate.dense.weight",
                f"encoder.layer.{i}.intermediate.dense.bias",
                f"encoder.layer.{i}.output.dense.weight",
                f"encoder.layer.{i}.output.dense.bias",
                f"encoder.layer.{i}.output.LayerNorm.weight",
                f"encoder.layer.{i}.output.LayerNorm.bias",
            ]
            mappings.extend(layer_mappings)
        
        model_mappings = [StateDictNameMapping(hf_name, paddle_name, hf_name) for paddle_name in mappings]
        
        # Initialize name mappings
        init_name_mappings(model_mappings)
        
        return model_mappings


class ModernBertForQuestionAnswering(ModernBertPretrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.modernbert = ModernBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.modernbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            
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


class ModernBertForMultipleChoice(ModernBertPretrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)

        self.modernbert = ModernBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else None

        input_ids = input_ids.reshape([-1, input_ids.shape[-1]]) if input_ids is not None else None
        attention_mask = attention_mask.reshape([-1, attention_mask.shape[-1]]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape([-1, token_type_ids.shape[-1]]) if token_type_ids is not None else None
        position_ids = position_ids.reshape([-1, position_ids.shape[-1]]) if position_ids is not None else None

        outputs = self.modernbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape([-1, num_choices])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ModernBertLMPredictionHead(Layer):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = F.gelu
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ModernBertForMaskedLM(ModernBertPretrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `ModernBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.modernbert = ModernBertModel(config)
        self.cls = ModernBertLMPredictionHead(config)

    def get_output_embeddings(self):
        return self.cls.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.modernbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.reshape([-1, self.config.vocab_size]), labels.reshape([-1]))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
