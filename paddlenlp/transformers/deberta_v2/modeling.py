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

import math

from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model

from ...utils.converter import StateDictNameMapping
from ...utils.env import CONFIG_NAME
from ..activations import ACT2FN
from ..model_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from .configuration import (
    DEBERTA_V2_PRETRAINED_INIT_CONFIGURATION,
    DEBERTA_V2_PRETRAINED_RESOURCE_FILES_MAP,
    DebertaV2Config,
)

__all__ = [
    "DebertaV2Model",
    "DebertaV2ForSequenceClassification",
    "DebertaV2ForQuestionAnswering",
    "DebertaV2ForTokenClassification",
    "DebertaV2PreTrainedModel",
    "DebertaV2ForMultipleChoice",
]
from collections.abc import Sequence

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def softmax_with_mask(x, mask, axis):
    rmask = paddle.logical_not(mask.astype("bool"))
    y = paddle.full(x.shape, -float("inf"), x.dtype)
    return F.softmax(paddle.where(rmask, y, x), axis=axis)


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        # mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)
        probability_matrix = paddle.full(paddle.empty_like(input).shape, 1 - dropout)
        mask = (1 - paddle.bernoulli(probability_matrix)).cast("bool")

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(paddle.autograd.PyLayer):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensor()
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(nn.Layer):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (`paddle.Tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


class GELUActivation(nn.Layer):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        self.act = nn.functional.gelu

    def _gelu_python(self, input):
        return input * 0.5 * (1.0 + paddle.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        return self.act(input)


class DebertaV2Embeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()

        pad_token_id = getattr(config, "pad_token_id", 0)
        self.position_biased_input = getattr(config, "position_biased_input", True)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)
        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias_attr=False)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = paddle.arange(seq_length, dtype="int64")
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            position_embeds = self.position_embeddings(position_ids)
        else:
            position_embeds = paddle.zeros_like(inputs_embeds)
        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings = embeddings + position_embeds
        if self.config.type_vocab_size > 0:
            token_type_embeds = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeds
        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)
        embeddings = self.LayerNorm(embeddings)
        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            embeddings = embeddings * mask.astype(embeddings.dtype)
        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaV2SelfOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Attention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        self_output = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )

        if output_attentions:
            self_output, att_matrix = self_output

        if query_states is None:
            query_states = hidden_states

        attention_output = self.output(self_output, query_states)

        if output_attentions:
            return (attention_output, att_matrix)
        else:
            return attention_output


class DebertaV2Intermediate(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = GELUActivation()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DebertaV2Output(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Layer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return layer_output


class ConvLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        self.conv = nn.Conv1D(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=groups,
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.transpose([0, 2, 1]))
        out = out.transpose([0, 2, 1])
        rmask = (1 - input_mask).astype(bool)
        mask = rmask.unsqueeze(-1).tile([1, 1, out.shape[2]])
        out = paddle.where(mask, paddle.zeros_like(out), out)
        out = GELUActivation()(self.dropout(out))

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input)

        if input_mask is None:
            output_states = output
        else:
            if input_mask.ndim != layer_norm_input.ndim:
                if input_mask.ndim == 4:
                    input_mask = paddle.squeeze(input_mask, [1, 2])
                input_mask = input_mask.unsqueeze(2)

            input_mask = input_mask.astype(output.dtype)
            output_states = output * input_mask

        return output_states


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    relative_pos = relative_pos.astype("float32")
    sign = paddle.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = paddle.where(
        (relative_pos < mid) & (relative_pos > -mid),
        paddle.to_tensor(mid - 1).astype(relative_pos.dtype),
        paddle.abs(relative_pos),
    )
    log_pos = (
        paddle.ceil(paddle.log(abs_pos / mid) / paddle.log(paddle.to_tensor((max_position - 1) / mid)) * (mid - 1))
        + mid
    )
    bucket_pos = paddle.where(abs_pos <= mid, relative_pos.astype(log_pos.dtype), log_pos * sign)
    return bucket_pos


def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
    """
    Build relative position according to the query and key

    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)

    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position

    Return:
        `paddle.Tensor`: A tensor with shape [1, query_size, key_size]
    """

    q_ids = paddle.arange(0, query_size, dtype="int64")
    k_ids = paddle.arange(0, key_size, dtype="int64")
    rel_pos_ids = q_ids.unsqueeze(1) - k_ids.unsqueeze(0)
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.astype("int64")
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return paddle.expand(
        c2p_pos, [query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], relative_pos.shape[-1]]
    )


def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return paddle.expand(
        c2p_pos, [query_layer.shape[0], query_layer.shape[1], key_layer.shape[-2], key_layer.shape[-2]]
    )


def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return paddle.expand(pos_index, p2c_att.shape[:2] + (pos_index.shape[-2], key_layer.shape[-2]))


class DisentangledSelfAttention(nn.Layer):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias_attr=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias_attr=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias_attr=True)

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias_attr=True)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.shape[:-1] + [attention_heads, -1]
        x = x.reshape(new_x_shape)
        return x.transpose(perm=[0, 2, 1, 3]).reshape([-1, x.shape[1], x.shape[-1]])

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = paddle.sqrt(paddle.to_tensor(query_layer.shape[-1], dtype=paddle.float32) * scale_factor)
        attention_scores = paddle.bmm(query_layer, key_layer.transpose([0, 2, 1])) / scale.astype(
            dtype=query_layer.dtype
        )
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = attention_scores
        attention_scores = attention_scores.reshape(
            [-1, self.num_attention_heads, attention_scores.shape[-2], attention_scores.shape[-1]]
        )

        # bsz x height x length x dimension
        attention_probs = softmax_with_mask(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        context_layer = paddle.bmm(
            attention_probs.reshape([-1, attention_probs.shape[-2], attention_probs.shape[-1]]), value_layer
        )
        context_layer = context_layer.reshape(
            [-1, self.num_attention_heads, context_layer.shape[-2], context_layer.shape[-1]]
        ).transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [
            -1,
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)
        if output_attentions:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.shape[-2]
            relative_pos = build_relative_position(
                q,
                key_layer.shape[-2],
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        if relative_pos.ndim == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.ndim == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bsz x height x query x key
        elif relative_pos.ndim != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.ndim}")

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.astype("int64")

        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(
                self.query_proj(rel_embeddings), self.num_attention_heads
            ).tile([query_layer.shape[0] // self.num_attention_heads, 1, 1])
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads).tile(
                [query_layer.shape[0] // self.num_attention_heads, 1, 1]
            )
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings), self.num_attention_heads
                ).tile([query_layer.shape[0] // self.num_attention_heads, 1, 1])
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings), self.num_attention_heads
                ).tile([query_layer.shape[0] // self.num_attention_heads, 1, 1])

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = paddle.sqrt(paddle.to_tensor(pos_key_layer.shape[-1], dtype=paddle.float32) * scale_factor)
            c2p_att = paddle.bmm(query_layer, pos_key_layer.transpose([0, 2, 1]))
            c2p_pos = paddle.clip(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = paddle.take_along_axis(
                c2p_att,
                axis=-1,
                indices=c2p_pos.squeeze(0).expand(
                    [query_layer.shape[0], query_layer.shape[1], relative_pos.shape[-1]]
                ),
            )
            score += c2p_att / scale.astype(dtype=c2p_att.dtype)
        # position->content
        if "p2c" in self.pos_att_type:
            scale = paddle.sqrt(paddle.to_tensor(pos_query_layer.shape[-1], dtype=paddle.float32) * scale_factor)
            if key_layer.shape[-2] != query_layer.shape[-2]:
                r_pos = build_relative_position(
                    key_layer.shape[-2],
                    key_layer.shape[-2],
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = paddle.clip(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = paddle.bmm(key_layer, pos_query_layer.transpose([0, 2, 1]))
            p2c_att = paddle.take_along_axis(
                p2c_att,
                axis=-1,
                indices=p2c_pos.squeeze(0).expand([query_layer.shape[0], key_layer.shape[-2], key_layer.shape[-2]]),
            ).transpose([0, 2, 1])
            score += p2c_att / scale.astype(dtype=p2c_att.dtype)

        return score


class DebertaV2Encoder(nn.Layer):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()

        self.layer = nn.LayerList([DebertaV2Layer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, bias_attr=True, weight_attr=True)

        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.astype(paddle.int8)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.shape[-2] if query_states is not None else hidden_states.shape[-2]
            relative_pos = build_relative_position(
                q,
                hidden_states.shape[-2],
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        return relative_pos

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=None,
    ):

        if attention_mask.ndim <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).astype(paddle.int8)
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            output_states = layer_module(
                next_kv,
                attention_mask,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
            )

            if output_attentions:
                output_states, att_m = output_states

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=output_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class DebertaV2PreTrainedModel(PretrainedModel):
    """
    An abstract class for pretrained BERT models. It provides BERT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = CONFIG_NAME
    config_class = DebertaV2Config
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "deberta"

    pretrained_init_configuration = DEBERTA_V2_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = DEBERTA_V2_PRETRAINED_RESOURCE_FILES_MAP

    @classmethod
    def _get_name_mappings(cls, config):
        mappings = []
        model_mappings = [
            ["embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight"],
            ["embeddings.LayerNorm.weight", "embeddings.LayerNorm.weight"],
            ["embeddings.LayerNorm.bias", "embeddings.LayerNorm.bias"],
            ["embeddings.position_embeddings.weight", "embeddings.position_embeddings.weight"],
            ["encoder.rel_embeddings.weight", "encoder.rel_embeddings.weight"],
            ["encoder.LayerNorm.weight", "encoder.LayerNorm.weight"],
            ["encoder.LayerNorm.bias", "encoder.LayerNorm.bias"],
        ]
        for layer_index in range(config.num_hidden_layers):

            layer_mappings = [
                [
                    f"encoder.layer.{layer_index}.attention.self.query_proj.weight",
                    f"encoder.layer.{layer_index}.attention.self.query_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.query_proj.bias",
                    f"encoder.layer.{layer_index}.attention.self.query_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.key_proj.weight",
                    f"encoder.layer.{layer_index}.attention.self.key_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.key_proj.bias",
                    f"encoder.layer.{layer_index}.attention.self.key_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.value_proj.weight",
                    f"encoder.layer.{layer_index}.attention.self.value_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.value_proj.bias",
                    f"encoder.layer.{layer_index}.attention.self.value_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.dense.weight",
                    f"encoder.layer.{layer_index}.attention.output.dense.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.dense.bias",
                    f"encoder.layer.{layer_index}.attention.output.dense.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.LayerNorm.weight",
                    f"encoder.layer.{layer_index}.attention.output.LayerNorm.weight",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.LayerNorm.bias",
                    f"encoder.layer.{layer_index}.attention.output.LayerNorm.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.intermediate.dense.weight",
                    f"encoder.layer.{layer_index}.intermediate.dense.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.intermediate.dense.bias",
                    f"encoder.layer.{layer_index}.intermediate.dense.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.output.dense.weight",
                    f"encoder.layer.{layer_index}.output.dense.weight",
                    "transpose",
                ],
                [f"encoder.layer.{layer_index}.output.dense.bias", f"encoder.layer.{layer_index}.output.dense.bias"],
                [
                    f"encoder.layer.{layer_index}.output.LayerNorm.weight",
                    f"encoder.layer.{layer_index}.output.LayerNorm.weight",
                ],
                [
                    f"encoder.layer.{layer_index}.output.LayerNorm.bias",
                    f"encoder.layer.{layer_index}.output.LayerNorm.bias",
                ],
            ]
            model_mappings.extend(layer_mappings)
        # adapt for hf-tiny-model-private/tiny-random-DebertaV2Model
        if config.architectures is not None and "DebertaV2Model" in config.architectures:
            pass
        else:
            for mapping in model_mappings:
                mapping[0] = "deberta." + mapping[0]
                mapping[1] = "deberta." + mapping[1]
        if config.architectures is not None and "DebertaV2ForQuestionAnswering" in config.architectures:
            model_mappings.extend(
                [["qa_outputs.weight", "qa_outputs.weight", "transpose"], ["qa_outputs.bias", "qa_outputs.bias"]]
            )
        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )

        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = self.config.layer_norm_eps


@register_base_model
class DebertaV2Model(DebertaV2PreTrainedModel):
    def __init__(self, config: DebertaV2Config):
        super(DebertaV2Model, self).__init__(config)
        self.config = config
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)
        self.z_steps = getattr(config, "z_steps", 0)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.ones(input_shape, dtype="int64")
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        if not return_dict:
            encoded_layers = encoder_outputs[1]
        else:
            encoded_layers = encoder_outputs.hidden_states

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )


class DebertaV2PredictionHeadTransform(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class DebertaV2LMPredictionHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.transform = DebertaV2PredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        self.bias = paddle.create_parameter(
            shape=[config.vocab_size], default_initializer=nn.initializer.Constant(0.0), dtype="float32"
        )
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class DebertaV2OnlyMLMHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.predictions = DebertaV2LMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class DebertaV2ForMaskedLM(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.cls = DebertaV2OnlyMLMHead(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deberta(
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
        prediction_scores = self.cls(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.reshape(-1, self.config.vocab_size), labels.reshape(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ContextPooler(nn.Layer):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.pooler_hidden_size if config.pooler_hidden_size is not None else config.hidden_size
        self.dense = nn.Linear(config.hidden_size, hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        context_token = hidden_states[:, 0, :]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = F.gelu(pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


class DebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim if self.pooler is not None else config.hidden_size
        self.classifier = nn.Linear(output_dim, config.num_labels)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.dropout = StableDropout(drop_out)

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = self.pooler(outputs[0])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = paddle.nn.MSELoss()
                loss = loss_fct(logits, labels)
            elif labels.dtype == paddle.int64 or labels.dtype == paddle.int32:
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1,)))
            else:
                loss_fct = paddle.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaV2ForTokenClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1,)))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaV2ForQuestionAnswering(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaV2Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if start_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = paddle.nn.CrossEntropyLoss(ignore_index=ignored_index)
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


class DebertaV2ForMultipleChoice(DebertaV2PreTrainedModel):

    """
    Deberta Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks.

    Args:
        bert (:class:`DebertaModel`):
            An instance of DebertaModel.
        num_choices (int, optional):
            The number of choices. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of Bert.
            If None, use the same value as `hidden_dropout_prob` of `DebertaModel`
            instance `bert`. Defaults to None.
    """

    def __init__(self, config: DebertaV2Config):
        super(DebertaV2ForMultipleChoice, self).__init__(config)
        self.deberta = DebertaV2Model(config)
        self.dropout = StableDropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.pooler = ContextPooler(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):
        r"""
        The DebertaForMultipleChoice forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`DebertaModel` and shape as [batch_size, num_choice, sequence_length].
            token_type_ids(Tensor, optional):
                See :class:`DebertaModel` and shape as [batch_size, num_choice, sequence_length].
            position_ids(Tensor, optional):
                See :class:`DebertaModel` and shape as [batch_size, num_choice, sequence_length].
            attention_mask (list, optional):
                See :class:`DebertaModel` and shape as [batch_size, num_choice, sequence_length].
            inputs_embeds (list, optional):
                See :class:`DebertaModel` and shape as [batch_size, num_choice, sequence_length].
            labels (Tensor of shape `(batch_size, )`, optional):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.MultipleChoiceModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.MultipleChoiceModelOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.model_outputs.MultipleChoiceModelOutput`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import BertForMultipleChoice, BertTokenizer
                from paddlenlp.data import Pad, Dict

                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model = BertForMultipleChoice.from_pretrained('bert-base-uncased', num_choices=2)

                data = [
                    {
                        "question": "how do you turn on an ipad screen?",
                        "answer1": "press the volume button.",
                        "answer2": "press the lock button.",
                        "label": 1,
                    },
                    {
                        "question": "how do you indent something?",
                        "answer1": "leave a space before starting the writing",
                        "answer2": "press the spacebar",
                        "label": 0,
                    },
                ]

                text = []
                text_pair = []
                for d in data:
                    text.append(d["question"])
                    text_pair.append(d["answer1"])
                    text.append(d["question"])
                    text_pair.append(d["answer2"])

                inputs = tokenizer(text, text_pair)
                batchify_fn = lambda samples, fn=Dict(
                    {
                        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
                        "token_type_ids": Pad(
                            axis=0, pad_val=tokenizer.pad_token_type_id
                        ),  # token_type_ids
                    }
                ): fn(samples)
                inputs = batchify_fn(inputs)

                reshaped_logits = model(
                    input_ids=paddle.to_tensor(inputs[0], dtype="int64"),
                    token_type_ids=paddle.to_tensor(inputs[1], dtype="int64"),
                )
                print(reshaped_logits.shape)
                # [2, 2]

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None:
            num_choices = input_ids.shape[1]
        elif inputs_embeds is not None:
            num_choices = inputs_embeds.shape[1]

        input_ids = input_ids.reshape((-1, input_ids.shape[-1])) if input_ids is not None else None
        inputs_embeds = (
            inputs_embeds.reshape((-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1]))
            if inputs_embeds is not None
            else None
        )
        position_ids = position_ids.reshape((-1, position_ids.shape[-1])) if position_ids is not None else None
        token_type_ids = token_type_ids.reshape((-1, token_type_ids.shape[-1])) if token_type_ids is not None else None
        attention_mask = attention_mask.reshape((-1, attention_mask.shape[-1])) if attention_mask is not None else None

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = self.pooler(outputs[0])
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape((-1, num_choices))

        loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
