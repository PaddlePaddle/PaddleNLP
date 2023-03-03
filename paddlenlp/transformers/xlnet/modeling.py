# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
"""Modeling classes for XLNet model."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, Layer, MSELoss

from ...utils.env import CONFIG_NAME
from .. import PretrainedModel, register_base_model
from ..activations import ACT2FN, get_activation
from ..model_outputs import ModelOutput, tuple_output
from .configuration import (
    XLNET_PRETRAINED_INIT_CONFIGURATION,
    XLNET_PRETRAINED_RESOURCE_FILES_MAP,
    XLNetConfig,
)

__all__ = [
    "XLNetPretrainedModel",
    "XLNetModel",
    "XLNetForSequenceClassification",
    "XLNetForTokenClassification",
    "XLNetLMHeadModel",
    "XLNetForMultipleChoice",
    "XLNetForQuestionAnswering",
    "XLNetForCausalLM",
]

dtype_float = paddle.get_default_dtype()


class XLNetRelativeAttention(Layer):
    def __init__(self, config: XLNetConfig):
        super(XLNetRelativeAttention, self).__init__()

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head**0.5)

        self.q = self.create_parameter([self.d_model, self.n_head * self.d_head])
        self.k = self.create_parameter([self.d_model, self.n_head * self.d_head])
        self.v = self.create_parameter([self.d_model, self.n_head * self.d_head])
        self.o = self.create_parameter([self.d_model, self.n_head * self.d_head])
        self.r = self.create_parameter([self.d_model, self.n_head * self.d_head])

        self.r_r_bias = self.create_parameter([self.n_head, self.d_head], is_bias=True)
        self.r_s_bias = self.create_parameter([self.n_head, self.d_head], is_bias=True)
        self.r_w_bias = self.create_parameter([self.n_head, self.d_head], is_bias=True)
        self.seg_embed = self.create_parameter([2, self.n_head, self.d_head], is_bias=False)

        self.layer_norm = nn.LayerNorm(config.d_model, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        # Relative shift of the attention matrix from bd~ to bd (refer to Appendix B in the Transformer-XL paper)
        x_size = paddle.shape(x)

        x = paddle.reshape(x, [x_size[0], x_size[1], x_size[3], x_size[2]])
        x = x[:, :, 1:, :]
        x = paddle.reshape(x, [x_size[0], x_size[1], x_size[2], x_size[3] - 1])
        x = paddle.index_select(x, index=paddle.arange(klen, dtype="int64"), axis=3)
        return x

    def rel_attn_core(
        self,
        q_head,
        k_head_h,
        v_head_h,
        k_head_r,
        seg_mat=None,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        """Core relative positional attention operations."""

        # Content based attention score (refer to the Transformer-XL paper)
        # q_head = Exi * Wq; self.r_w_bias = u; k_head_h = Wke * Exj
        # a = Exi * Wq * Wke * Exj; c = u * Wke * Exj; ac = a + c
        ac = paddle.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)

        # Position based attention score (refer to the Transformer-XL paper)
        # q_head = Exi * Wq; self.r_r_bias = v; k_head_r = Wkr * Rij
        # b = Exi * Wq * Wkr * Rij; d = v * Wkr * Rij; bd = b + d
        bd = paddle.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=paddle.shape(ac)[3])

        # Segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = paddle.einsum("ibnd,snd->ibns", q_head + self.r_s_bias, self.seg_embed)
            ef = paddle.einsum("ijbs,ibns->bnij", seg_mat, ef)

        # Merge attention scores and perform masking
        attn_score = (ac + bd + ef) * self.scale

        if attn_mask is not None:
            attn_mask = attn_mask.transpose([2, 3, 0, 1])
            attn_score = attn_score - 1e30 * attn_mask

        # Attention probability
        attn_prob = F.softmax(attn_score, axis=3)
        attn_prob = self.dropout(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask.transpose([2, 3, 0, 1])

        # Attention output
        attn_vec = paddle.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)

        if output_attentions:
            return attn_vec, attn_prob.transpose([2, 3, 0, 1])
        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        # Post-attention projection (back to 'd_model')
        # Compute einsum4x4("ibnd,hnd->ibh", attn_vec, self.o)
        shape = paddle.shape(attn_vec)
        attn_vec = attn_vec.reshape([shape[0], shape[1], attn_vec.shape[2] * attn_vec.shape[3]])
        attn_out = paddle.einsum("ibm,hm->ibh", attn_vec, self.o)

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h

        output = self.layer_norm(attn_out)
        return output

    def forward(
        self,
        h,
        g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems=None,
        target_mapping=None,
        head_mask=None,
        output_attentions=False,
    ):
        if g is not None:
            # Two-stream attention with relative positional encoding.
            # Content based attention score
            if mems is not None and mems.dim() > 1:
                cat = paddle.concat([mems, h], axis=0)
            else:
                cat = h

            # Content-based key head
            # Compute k_head_h = einsum4x4("ibh,h(n*d)->ibnd", cat, self.k)
            k_head_h = paddle.matmul(cat, self.k)
            k_head_h = paddle.reshape(
                k_head_h, shape=[paddle.shape(cat)[0], paddle.shape(cat)[1], self.n_head, self.d_head]
            )

            # Content-based value head
            # Compute v_head_h = einsum4x4("ibh,h(n*d)->ibnd", cat, self.v)
            v_head_h = paddle.matmul(cat, self.v)
            v_head_h = paddle.reshape(
                v_head_h, shape=[paddle.shape(cat)[0], paddle.shape(cat)[1], self.n_head, self.d_head]
            )

            # Position-based key head
            # Compute k_head_r = einsum4x4("ibh,h(n*d)->ibnd", r, self.r)
            k_head_r = paddle.matmul(r, self.r)
            k_head_r = paddle.reshape(
                k_head_r, shape=[paddle.shape(r)[0], paddle.shape(r)[1], self.n_head, self.d_head]
            )

            # H-stream
            # Content-stream query head
            # Compute q_head_h = einsum4x4("ibh,h(n*d)->ibnd", h, self.q)
            q_head_h = paddle.matmul(h, self.q)  # shape
            q_head_h = paddle.reshape(
                q_head_h, shape=[paddle.shape(h)[0], paddle.shape(h)[1], self.n_head, self.d_head]
            )

            # Core attention ops
            attn_vec_h = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h

            # Post processing
            output_h = self.post_attention(h, attn_vec_h)

            # G-stream
            # Query-stream query head
            # Compute q_head_g = einsum4x4("ibh,hnd->ibnd", g, self.q)
            shape = g.shape
            q_head_g = paddle.matmul(g, self.q).reshape([shape[0], shape[1], self.n_head, self.d_head])

            # Core attention ops
            if target_mapping is not None:
                # Compute q_head_g = einsum4x4("mbnd,mlb->lbnd", q_head_g, target_mapping)
                q_head_g = paddle.einsum("mbnd,mlb->lbnd", q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                )

                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

                # Compute attn_vec_g = einsum4x4("lbnd,mlb->mbnd", attn_vec_g, target_mapping)
                attn_vec_g = paddle.einsum("lbnd,mlb->mbnd", attn_vec_g, target_mapping)

            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                )

                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

            # Post processing
            output_g = self.post_attention(g, attn_vec_g)

            if output_attentions:
                attn_prob = attn_prob_h, attn_prob_g

        else:
            # Multi-head attention with relative positional encoding
            if mems is not None and mems.dim() > 1:
                cat = paddle.concat([mems, h], axis=0)
            else:
                cat = h

            # Content heads
            # Compute q_head_h = einsum4x4("ibh,hnd->ibnd", h, self.q)
            q_head_h = paddle.matmul(h, self.q)
            q_head_h = paddle.reshape(
                q_head_h, shape=[paddle.shape(h)[0], paddle.shape(h)[1], self.n_head, self.d_head]
            )

            # Compute k_head_h = einsum4x4("ibh,hnd->ibnd", cat, self.k)
            k_head_h = paddle.matmul(cat, self.k)
            k_head_h = paddle.reshape(
                k_head_h, shape=[paddle.shape(h)[0], paddle.shape(h)[1], self.n_head, self.d_head]
            )

            # Compute v_head_h = einsum4x4("ibh,hnd->ibnd", cat, self.v)
            v_head_h = paddle.matmul(cat, self.v)
            v_head_h = paddle.reshape(
                v_head_h, shape=[paddle.shape(h)[0], paddle.shape(h)[1], self.n_head, self.d_head]
            )

            # Position-based key head
            # Compute k_head_r = einsum4x4("ibh,hnd->ibnd", r, self.r)
            k_head_r = paddle.matmul(r, self.r)
            k_head_r = paddle.reshape(k_head_r, shape=[paddle.shape(k_head_r)[0], -1, self.n_head, self.d_head])

            # Core attention ops
            attn_vec = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                attn_vec, attn_prob = attn_vec

            # Post processing
            output_h = self.post_attention(h, attn_vec)
            output_g = None

        outputs = (output_h, output_g)

        if output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs


class XLNetFeedForward(Layer):
    def __init__(self, config: XLNetConfig):
        super(XLNetFeedForward, self).__init__()

        self.layer_norm = nn.LayerNorm(config.d_model, epsilon=config.layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if isinstance(config.ff_activation, str):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, inp):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output


class XLNetLayer(Layer):
    def __init__(self, config: XLNetConfig):
        super(XLNetLayer, self).__init__()

        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.seq_len_dim = 1

    def forward(
        self,
        output_h,
        output_g,
        attn_mask_h,
        attn_mask_g,
        r,
        seg_mat,
        mems=None,
        target_mapping=None,
        head_mask=None,
        output_attentions=False,
    ):
        outputs = self.rel_attn(
            output_h,
            output_g,
            attn_mask_h,
            attn_mask_g,
            r,
            seg_mat,
            mems=mems,
            target_mapping=target_mapping,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        output_h, output_g = outputs[:2]

        if output_g is not None:
            output_g = self.ff(output_g)
        output_h = self.ff(output_h)

        outputs = (output_h, output_g) + outputs[2:]  # Add again attentions if they are there
        return outputs


class XLNetPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained XLNet models. It provides XLNet related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading
    and loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    pretrained_init_configuration = XLNET_PRETRAINED_INIT_CONFIGURATION
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = XLNET_PRETRAINED_RESOURCE_FILES_MAP
    model_config_file = CONFIG_NAME
    config_class = XLNetConfig
    base_model_prefix = "transformer"

    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        # Initialize the weights.
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range")
                        else self.transformer.config["initializer_range"],
                        shape=layer.weight.shape,
                    )
                )
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.full_like(layer.weight, 1.0))
        elif isinstance(layer, XLNetRelativeAttention):
            for param in [
                layer.q,
                layer.k,
                layer.v,
                layer.o,
                layer.r,
                layer.r_r_bias,
                layer.r_s_bias,
                layer.r_w_bias,
                layer.seg_embed,
            ]:
                param.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range")
                        else self.transformer.config["initializer_range"],
                        shape=param.shape,
                    )
                )
        elif isinstance(layer, XLNetModel):
            layer.mask_emb.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range")
                    else self.transformer.config["initializer_range"],
                    shape=layer.mask_emb.shape,
                )
            )


@dataclass
class XLNetModelOutput(ModelOutput):
    """
    Output type of [`XLNetModel`].

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        mems (`List[paddle.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: paddle.Tensor
    mems: Optional[List[paddle.Tensor]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class XLNetLMHeadModelOutput(ModelOutput):
    """
    Output type of [`XLNetLMHeadModel`].

    Args:
        loss (`paddle.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided)
            Language modeling loss (for next-token prediction).
        logits (`paddle.Tensor` of shape `(batch_size, num_predict, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        mems (`List[paddle.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    mems: Optional[List[paddle.Tensor]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class XLNetForSequenceClassificationOutput(ModelOutput):
    """
    Output type of [`XLNetForSequenceClassification`].

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`paddle.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (`List[paddle.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    mems: Optional[List[paddle.Tensor]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class XLNetForTokenClassificationOutput(ModelOutput):
    """
    Output type of [`XLNetForTokenClassificationOutput`].

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        mems (`List[paddle.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    mems: Optional[List[paddle.Tensor]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class XLNetForMultipleChoiceOutput(ModelOutput):
    """
    Output type of [`XLNetForMultipleChoice`].

    Args:
        loss (`paddle.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`paddle.Tensor` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

            Classification scores (before SoftMax).
        mems (`List[paddle.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    mems: Optional[List[paddle.Tensor]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class XLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    """
    Output type of [`XLNetForQuestionAnsweringSimple`].

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`paddle.Tensor` of shape `(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_logits (`paddle.Tensor` of shape `(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        mems (`List[paddle.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    start_logits: paddle.Tensor = None
    end_logits: paddle.Tensor = None
    mems: Optional[List[paddle.Tensor]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class XLNetForQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`XLNetForQuestionAnswering`].

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (`paddle.Tensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (`paddle.Tensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (`paddle.Tensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
            (beam-search).
        end_top_index (`paddle.Tensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        cls_logits (`paddle.Tensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the `is_impossible` label of the answers.
        mems (`List[paddle.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    start_top_log_probs: Optional[paddle.Tensor] = None
    start_top_index: Optional[paddle.Tensor] = None
    end_top_log_probs: Optional[paddle.Tensor] = None
    end_top_index: Optional[paddle.Tensor] = None
    cls_logits: Optional[paddle.Tensor] = None
    mems: Optional[List[paddle.Tensor]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@register_base_model
class XLNetModel(XLNetPretrainedModel):
    """
    The bare XLNet Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `XLNetModel`.
            Also is the vocab size of token embedding matrix.
        mem_len (int or None, optional):
            The number of tokens to cache. If not 0 or None, the last `mem_len` hidden states
            in each layer will be cached into memory. Defaults to `None`.
        reuse_len (int or None, optional):
            The number of tokens in the current batch to be cached. If positive, then at most
            `reuse_len` tokens can be cached in the current batch. Otherwise, there is
            no limit to the number of tokens. Defaults to `None`.

            .. note::
                The difference between `mem_len` and `reuse_len` is that `mem_len` defines
                **the total number** of tokens to cache while `reuse_len` defines the number of tokens
                in **the current batch** to be cached.
        d_model (int, optional):
            Dimensionality of the embedding layers, encoder layers and pooler layer.
            Defaults to 768.
        same_length (bool, optional):
            Whether or not to use the same attention length for each token.
            Defaults to `False`.
        attn_type (str, optional):
            The attention type used in the attention layer. Set **"bi"** for ``XLNet``,
            **"uni"** for ``Transformer-XL``. Defaults to **"bi"**.
        bi_data (bool, optional):
            Whether or not to use bidirectional input pipeline. Set to `True` during pretraining and
            `False` during fine-tuning. Defaults to `False`.
        clamp_len (int, optional):
            Maximum relative distance supported. All relative distances larger than `clamp_len` will be clamped.
            Setting this attribute to -1 means no clamping. Defaults to -1.
        n_layer (int, optional):
            The number of hidden layers in the encoder. Defaults to 12.
        dropout (float, optional):
            The dropout ratio for all fully connected layers in the embeddings and encoder.
            Defaults to 0.1.
        classifier_dropout (float, optional):
            The dropout ratio for all fully connected layers in the pooler (classification head).
            Defaults to 0.1.
        n_head (int, optional):
            Number of attention heads in each attention layer.
            Defaults to 12.
        d_head (int, optional):
            Dimensionality of each attention head. Defaults to 64.

            .. note::
                `d_head` should be equal to `d_model` divided by `n_head`.
        layer_norm_eps (float, optional):
            The `epsilon` parameter used in :class:`paddle.nn.LayerNorm` for
            initializing layer normalization layers. Defaults to 1e-12.
        d_inner (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `d_model` to `d_inner`,
            and then projected back to `d_model`. Typically `d_inner` is larger than `d_model`.
            Defaults to 3072.
        ff_activation (str, optional):
            The non-linear activation function in the feed-forward layers in the encoder.
            Choose from the following supported activation functions: `["relu", "gelu", "tanh",
            "sigmoid", "mish", "swish"]`. Defaults to `"gelu"`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to 0.02.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`XLNetPretrainedModel._init_weights()` for how weights are initialized in `XLNetModel`.
    """

    def __init__(self, config: XLNetConfig):
        super(XLNetModel, self).__init__(config)
        self.initializer_range = config.initializer_range
        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer
        self.dropout = nn.Dropout(config.dropout)
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = self.create_parameter([1, 1, config.d_model])
        self.layer = nn.LayerList([XLNetLayer(config) for _ in range(config.n_layer)])

        self.init_weights()

    def get_input_embeddings(self):
        return self.word_embedding

    def set_input_embeddings(self, new_embeddings):
        self.word_embedding = new_embeddings

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def create_mask(self, qlen, mlen):
        # Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.
        attn_mask = paddle.ones([qlen, qlen])
        mask_up = paddle.triu(attn_mask, diagonal=1)
        attn_mask_pad = paddle.zeros([qlen, mlen])
        ret = paddle.concat([attn_mask_pad, mask_up], axis=1)
        if self.same_length:
            mask_lo = paddle.tril(attn_mask, diagonal=-1)
            ret = paddle.concat([ret[:, :qlen] + mask_lo, ret[:, qlen:]], axis=1)

        return ret

    def cache_mem(self, curr_out, prev_mem):
        # Cache hidden states into memory.
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]

        if self.mem_len is None or self.mem_len == 0:
            # If `use_mems` is active but no `mem_len` is defined, the model behaves like GPT-2 at inference time
            # and returns all of the past and current hidden states.
            cutoff = 0
        else:
            # If :obj:`use_mems` is active and `mem_len` is defined, the model returns the last `mem_len` hidden
            # states. This is the preferred setting for training and long-form generation.
            cutoff = -self.mem_len
        if prev_mem is None:
            # If :obj:`use_mems` is active and `mem_len` is defined, the model
            new_mem = curr_out[cutoff:]
        else:
            new_mem = paddle.concat([prev_mem, curr_out], axis=0)[cutoff:]

        return new_mem.detach()

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        # Compute sinusoid_inp = einsum4x4("i,d->id", pos_seq, inv_freq)
        sinusoid_inp = paddle.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = paddle.concat([paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=-1)
        pos_emb = paddle.unsqueeze(pos_emb, axis=1)
        if bsz is not None:
            pos_emb = pos_emb.expand([-1, bsz, -1])
            pos_emb.stop_gradient = True
        pos_emb.stop_gradient = True
        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        # Create relative positional encoding.
        freq_seq = paddle.arange(0, self.d_model, 2.0, dtype=dtype_float)
        inv_freq = 1 / 10000 ** (freq_seq / self.d_model)

        if self.attn_type == "bi":
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            beg, end = klen, -1
        else:
            raise ValueError("Unknown `attn_type` {}.".format(self.attn_type))

        if self.bi_data:
            fwd_pos_seq = paddle.arange(beg, end, -1.0, dtype=dtype_float)
            bwd_pos_seq = paddle.arange(-beg, -end, 1.0, dtype=dtype_float)

            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)

            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)
            pos_emb = paddle.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
        else:
            fwd_pos_seq = paddle.arange(beg, end, -1.0, dtype=dtype_float)
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)
        return pos_emb

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_mems_train=False,
        use_mems_eval=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        The XLNetModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate first and second portions of the inputs.
                Indices can be either 0 or 1:

                - 0 corresponds to a **sentence A** token,
                - 1 corresponds to a **sentence B** token.

                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
                Defaults to None, which means no segment embeddings is added to token embeddings.
            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.

                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.

                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.
            mems (List[Tensor], optional):
                A list of length `n_layers` with each Tensor being a pre-computed hidden-state for each layer.
                Each Tensor has a dtype `float32` and a shape of [batch_size, sequence_length, hidden_size].
                Defaults to None, and we don't use mems.

                .. note::
                    `use_mems` has to be set to `True` in order to make use of `mems`.
            perm_mask (Tensor, optional):
                Mask to indicate the permutation pattern of the input sequence with values being either 0 or 1.

                - if ``perm_mask[k, i, j] = 0``, i **attend** to j in batch k;
                - if ``perm_mask[k, i, j] = 1``, i **does not attend** to j in batch k.

                Only used during pretraining (to define factorization order) or
                for sequential decoding (generation). It's data type should be `float32` and
                has a shape of [batch_size, sequence_length, sequence_length].
                Defaults to `None`, then each token attends to all the other tokens (full bidirectional attention).
            target_mapping (Tensor, optional):
                Mask to indicate the output tokens to use with values being either 0 or 1.
                If ``target_mapping[k, i, j] = 1``, the i-th predict in batch k is on the j-th token.
                It's data type should be `float32` and has a shape of [batch_size, num_predict, sequence_length].
                Only used during pretraining for partial prediction or for sequential decoding (generation).
                Defaults to `None`.
            input_mask (Tensor, optional):
                Mask to avoid performing attention on padding token with values being either 0 or 1.
                It's data type should be `float32` and it has a shape of [batch_size, sequence_length].
                This mask is negative of `attention_mask`:

                - 1 for tokens that are **masked**,
                - 0 for tokens that are **not masked**.

                You should use only one of `input_mask` and `attention_mask`. Defaults to `None`.
            head_mask (Tensor, optional):
                Mask to nullify selected heads of the self-attention layers with values being either 0 or 1.

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

                It's data type should be `float32` and has a shape of [num_heads] or [num_layers, num_heads].
                Defaults to `None`, which means we keep all heads.
            inputs_embeds (Tensor, optional):
                An embedded representation tensor which is an alternative of `input_ids`.
                You should specify only either one of them to avoid contradiction.
                It's data type should be `float32` and has a shape of [batch_size, sequence_length, hidden_size].
                Defaults to `None`, which means we only specify `input_ids`.
            use_mems_train (bool, optional):
                Whether or not to use recurrent memory mechanism during training.
                Defaults to `False` and we don't use recurrent memory mechanism in training mode.
            use_mems_eval (bool, optional):
                Whether or not to use recurrent memory mechanism during evaluation.
                Defaults to `False` and we don't use recurrent memory mechanism in evaluation mode.
            return_dict (bool, optional):
                Whether or not to return additional information other than the output tensor.
                If True, then returns information about `output`, `new_mems`, `hidden_states` and `attentions`
                which will also be formatted as a dict. Else only returns the output tensor.
                Defaults to False.

        Returns:
            Tensor or dict: Returns tensor `output` or a dict with key-value pairs:
            {"last_hidden_state": `output`, "mems": `mems`,
            "hidden_states": `hidden_states`, "attentions": `attentions`}.

            With the corresponding fields:

            - `output` (Tensor):
                Output of the final layer of the model.
                It's a Tensor of dtype `float32` and has a shape of [batch_size, num_predict, hidden_size].

                .. note::
                    `num_predict` corresponds to `target_mapping.shape[1]`.
                    If `target_mapping` is `None`, then `num_predict` equals to `sequence_length`.
            - `mems` (List[Tensor]):
                A list of pre-computed hidden-states. The length of the list is `n_layers`.
                Each element in the list is a Tensor with dtype `float32` and has a shape of
                [batch_size, sequence_length, hidden_size].
            - `hidden_states` (List[Tensor], optional):
                A list of Tensor containing hidden-states of the model at the output of each layer
                plus the initial embedding outputs. Each Tensor has a data type of `float32` and
                has a shape of [batch_size, sequence_length, hidden_size].
                Being returned when `output_hidden_states` is set to `True`.
            - `attentions` (List[Tensor], optional):
                A list of Tensor containing attentions weights of each hidden layer.
                Each Tensor (one for each layer) has a data type of `float32` and
                has a shape of [batch_size, num_heads, sequence_length, sequence_length].
                Being returned when `output_attentions` is set to `True`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.xlnet.modeling import XLNetModel
                from paddlenlp.transformers.xlnet.tokenizer import XLNetTokenizer

                tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
                model = XLNetModel.from_pretrained('xlnet-base-cased')

                inputs = tokenizer("Hey, Paddle-paddle is awesome !")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                last_hidden_states = outputs[0]
        """

        if self.training:
            use_mems = use_mems_train
        else:
            use_mems = use_mems_eval

        # The original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = paddle.transpose(input_ids, perm=[1, 0])
            qlen, bsz = paddle.shape(input_ids)[0], paddle.shape(input_ids)[1]
        elif inputs_embeds is not None:
            inputs_embeds = paddle.transpose(inputs_embeds, perm=[1, 0])
            qlen, bsz = paddle.shape(inputs_embeds)[0], paddle.shape(inputs_embeds)[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        token_type_ids = token_type_ids.transpose([1, 0]) if token_type_ids is not None else None
        input_mask = input_mask.transpose([1, 0]) if input_mask is not None else None
        attention_mask = attention_mask.transpose([1, 0]) if attention_mask is not None else None
        perm_mask = perm_mask.transpose([1, 2, 0]) if perm_mask is not None else None
        target_mapping = target_mapping.transpose([1, 2, 0]) if target_mapping is not None else None

        mlen = paddle.shape(mems[0])[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen

        # Attention mask
        # Causal attention mask
        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = paddle.unsqueeze(attn_mask, axis=[2, 3])
        elif self.attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError("Unsupported attention type: {}".format(self.attn_type))

        # Data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatibility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = paddle.unsqueeze(input_mask, axis=0) + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = paddle.unsqueeze(input_mask, axis=0)
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # All mems can be attended to
            if mlen > 0:
                mems_mask = paddle.cast(paddle.zeros([paddle.shape(data_mask)[0], mlen, bsz]), dtype=dtype_float)
                data_mask = paddle.concat([mems_mask, data_mask], axis=1)
            if attn_mask is None:
                attn_mask = paddle.unsqueeze(data_mask, axis=-1)
            else:
                attn_mask += paddle.unsqueeze(data_mask, axis=-1)

        if attn_mask is not None:
            attn_mask = paddle.cast((attn_mask > 0), dtype=dtype_float)

        if attn_mask is not None:
            fill_val = paddle.ones(qlen)
            non_tgt_mask = paddle.cast(-paddle.diag(fill_val), dtype=dtype_float)
            if mlen > 0:
                non_tgt_mask = paddle.concat(
                    [paddle.cast(paddle.zeros([qlen, mlen]), dtype=dtype_float), non_tgt_mask], axis=-1
                )
            non_tgt_mask = paddle.cast(
                ((attn_mask + paddle.unsqueeze(non_tgt_mask, axis=[2, 3])) > 0), dtype=dtype_float
            )
        else:
            non_tgt_mask = None

        # Word embeddings and prepare h & g hidden states
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            word_emb_k = self.word_embedding(input_ids)

        output_h = self.dropout(word_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand([paddle.shape(target_mapping)[0], bsz, -1])
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        # Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            if mlen > 0:
                mem_pad = paddle.zeros(shape=[mlen, bsz], dtype="int64")
                cat_ids = paddle.concat(x=[mem_pad, token_type_ids], axis=0)
            else:
                cat_ids = token_type_ids

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = paddle.cast(
                paddle.unsqueeze(token_type_ids, axis=1) != paddle.unsqueeze(cat_ids, axis=0), dtype="int64"
            )
            seg_mat = paddle.cast(F.one_hot(seg_mat, num_classes=2), dtype=dtype_float)
        else:
            seg_mat = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # Attention_probs has shape bsz x n_heads x N x N
        # Input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # And head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand([self.n_layer, -1, -1, -1, -1])
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = [] if output_attentions else None
        hidden_states = [] if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if use_mems:
                # Cache new mems
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)

            outputs = layer_module(
                output_h,
                output_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                seg_mat=seg_mat,
                mems=mems[i],
                target_mapping=target_mapping,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            output_h, output_g = outputs[:2]

            if output_attentions:
                attentions.append(outputs[2])

        # Add last hidden state
        if output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = paddle.transpose(output, perm=[1, 0, 2])

        if not use_mems:
            new_mems = None

        if output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(paddle.transpose(h, perm=[1, 0, 2]) for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(paddle.transpose(hs, perm=[1, 0, 2]) for hs in hidden_states)

        if output_attentions:
            if target_mapping is not None:
                # When target_mapping is provided, there are 2-tuple of attentions
                attentions = tuple(
                    tuple(paddle.transpose(att_stream, perm=[2, 3, 0, 1]) for att_stream in t) for t in attentions
                )
            else:
                attentions = tuple(paddle.transpose(t, perm=[2, 3, 0, 1]) for t in attentions)

        if not return_dict:
            return tuple(v for v in [output, new_mems, hidden_states, attentions] if v is not None)

        return XLNetModelOutput(
            last_hidden_state=output, mems=new_mems, hidden_states=hidden_states, attentions=attentions
        )


class XLNetClassificationHead(Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: XLNetConfig):
        super(XLNetClassificationHead, self).__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.classfier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, -1, :]  # Take <cls> token
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("tanh")(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class XLNetForSequenceClassification(XLNetPretrainedModel):
    """
    XLNet Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        xlnet (:class:`XLNetModel`):
            An instance of :class:`XLNetModel`.
        num_classes (int, optional):
            The number of classes. Defaults to 2.
    """

    def __init__(self, config: XLNetConfig):
        super(XLNetForSequenceClassification, self).__init__(config)
        self.num_classes = config.num_classes
        self.transformer = XLNetModel(config)
        self.classifier = XLNetClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_mems_train=False,
        use_mems_eval=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        problem_type: str = "single_label_classification",
    ):
        r"""
        The XLNetForSequenceClassification forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                See :class:`XLNetModel`.
            token_type_ids (Tensor, optional):
                See :class:`XLNetModel`.
            attention_mask (Tensor, optional):
                See :class:`XLNetModel`.
            mems (Tensor, optional):
                See :class:`XLNetModel`.
            perm_mask (Tensor, optional):
                See :class:`XLNetModel`.
            target_mapping (Tensor, optional):
                See :class:`XLNetModel`.
            input_mask (Tensor, optional):
                See :class:`XLNetModel`.
            head_mask (Tensor, optional):
                See :class:`XLNetModel`.
            inputs_embeds (Tensor, optional):
                See :class:`XLNetModel`.
            use_mems_train (bool, optional):
                See :class:`XLNetModel`.
            use_mems_eval (bool, optional):
                See :class:`XLNetModel`.
            return_dict (bool, optional):
                See :class:`XLNetModel`.

        Returns:
            Tensor or dict: Returns tensor `logits` or a dict with key-value pairs:
            {"logits": `logits`, "mems": `mems`,
            "hidden_states": `hidden_states`, "attentions": `attentions`}.

            With the corresponding fields:

            - `logits` (Tensor):
                Classification scores before SoftMax (also called logits). It's data type should be `float32`
                and has a shape of [batch_size, num_classes].
            - `mems` (List[Tensor]):
                See :class:`XLNetModel`.
            - `hidden_states` (List[Tensor], optional):
                See :class:`XLNetModel`.
            - `attentions` (List[Tensor], optional):
                See :class:`XLNetModel`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.xlnet.modeling import XLNetForSequenceClassification
                from paddlenlp.transformers.xlnet.tokenizer import XLNetTokenizer

                tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
                model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')

                inputs = tokenizer("Hey, Paddle-paddle is awesome !")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """

        transformer_outputs = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems_train=use_mems_train,
            use_mems_eval=use_mems_eval,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]

        logits = self.classifier(output)

        loss = None
        if labels is not None:

            if problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_classes == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.reshape(shape=[-1, self.num_classes]), labels.reshape(shape=[-1]))
            elif problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return tuple_output(output, loss)

        return XLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLNetForTokenClassification(XLNetPretrainedModel):
    """
    XLNet Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        xlnet (:class:`XLNetModel`):
            An instance of :class:`XLNetModel`.
        num_classes (int, optional):
            The number of classes. Defaults to 2.
    """

    def __init__(self, config: XLNetConfig):
        super(XLNetForTokenClassification, self).__init__(config)
        self.num_classes = config.num_classes

        self.transformer = XLNetModel(config)
        self.classifier = nn.Linear(self.transformer.d_model, config.num_classes)

        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_mems_train=False,
        use_mems_eval=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        The XLNetForTokenClassification forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                See :class:`XLNetModel`.
            token_type_ids (Tensor, optional):
                See :class:`XLNetModel`.
            attention_mask (Tensor, optional):
                See :class:`XLNetModel`.
            mems (Tensor, optional):
                See :class:`XLNetModel`.
            perm_mask (Tensor, optional):
                See :class:`XLNetModel`.
            target_mapping (Tensor, optional):
                See :class:`XLNetModel`.
            input_mask (Tensor, optional):
                See :class:`XLNetModel`.
            head_mask (Tensor, optional):
                See :class:`XLNetModel`.
            inputs_embeds (Tensor, optional):
                See :class:`XLNetModel`.
            use_mems_train (bool, optional):
                See :class:`XLNetModel`.
            use_mems_eval (bool, optional):
                See :class:`XLNetModel`.
            return_dict (bool, optional):
                See :class:`XLNetModel`.

        Returns:
            Tensor or dict: Returns tensor `logits` or a dict with key-value pairs:
             {"logits": `logits`, "mems": `mems`,
            "hidden_states": `hidden_states`, "attentions": `attentions`}.

            With the corresponding fields:

            - `logits` (Tensor):
                Classification scores before SoftMax (also called logits). It's data type should be `float32`
                and has a shape of [batch_size, sequence_length, num_classes].
            - `mems` (List[Tensor]):
                See :class:`XLNetModel`.
            - `hidden_states` (List[Tensor], optional):
                See :class:`XLNetModel`.
            - `attentions` (List[Tensor], optional):
                See :class:`XLNetModel`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.xlnet.modeling import XLNetForTokenClassification
                from paddlenlp.transformers.xlnet.tokenizer import XLNetTokenizer

                tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
                model = XLNetForTokenClassification.from_pretrained('xlnet-base-cased')

                inputs = tokenizer("Hey, Paddle-paddle is awesome !")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """
        outputs = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems_train=use_mems_train,
            use_mems_eval=use_mems_eval,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(shape=[-1, self.num_classes]), labels.reshape(shape=[-1]))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return tuple_output(output, loss)

        return XLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=outputs.mems,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class XLNetLMHeadModel(XLNetPretrainedModel):
    """
    XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).

    Args:
        xlnet (:class:`XLNetModel`):
            An instance of :class:`XLNetModel`.
    """

    def __init__(self, config: XLNetConfig):
        super(XLNetLMHeadModel, self).__init__(config)
        self.transformer = XLNetModel(config)
        self.decoder_weight = self.transformer.word_embedding.weight
        self.decoder_bias = self.create_parameter(
            shape=[self.transformer.config["vocab_size"]], dtype=self.decoder_weight.dtype, is_bias=True
        )
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_mems_train=False,
        use_mems_eval=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        The XLNetLMHeadModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                See :class:`XLNetModel`.
            token_type_ids (Tensor, optional):
                See :class:`XLNetModel`.
            attention_mask (Tensor, optional):
                See :class:`XLNetModel`.
            mems (Tensor, optional):
                See :class:`XLNetModel`.
            perm_mask (Tensor, optional):
                See :class:`XLNetModel`.
            target_mapping (Tensor, optional):
                See :class:`XLNetModel`.
            input_mask (Tensor, optional):
                See :class:`XLNetModel`.
            head_mask (Tensor, optional):
                See :class:`XLNetModel`.
            inputs_embeds (Tensor, optional):
                See :class:`XLNetModel`.
            use_mems_train (bool, optional):
                See :class:`XLNetModel`.
            use_mems_eval (bool, optional):
                See :class:`XLNetModel`.
            return_dict (bool, optional):
                See :class:`XLNetModel`.

        Returns:
            Tensor or dict: Returns tensor `logits` or a dict with key-value pairs:
             {"logits": `logits`, "mems": `mems`,
            "hidden_states": `hidden_states`, "attentions": `attentions`}.

            With the corresponding fields:

            - `logits` (Tensor):
                Classification scores before SoftMax (also called logits). It's data type should be `float32`
                and has a shape of [batch_size, sequence_length, num_classes].
            - `mems` (List[Tensor]):
                See :class:`XLNetModel`.
            - `hidden_states` (List[Tensor], optional):
                See :class:`XLNetModel`.
            - `attentions` (List[Tensor], optional):
                See :class:`XLNetModel`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.xlnet.modeling import XLNetLMHeadModel
                from paddlenlp.transformers.xlnet.tokenizer import XLNetTokenizer

                tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
                model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')

                inputs = tokenizer("Hey, Paddle-paddle is awesome !")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)
                logits = outputs
        """
        transformer_outputs = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems_train=use_mems_train,
            use_mems_eval=use_mems_eval,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = paddle.matmul(transformer_outputs[0], self.decoder_weight, transpose_y=True) + self.decoder_bias
        loss = None
        if labels is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(shape=[-1, logits.shape[-1]]), labels.reshape(shape=[-1]))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return tuple_output(output, loss)

        return XLNetLMHeadModelOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLNetForMultipleChoice(XLNetPretrainedModel):
    """
    XLNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RACE/SWAG tasks.

    Args:
        xlnet (:class:`XLNetModel`):
            An instance of :class:`XLNetModel`.
    """

    def __init__(self, config: XLNetConfig):
        super(XLNetForMultipleChoice, self).__init__(config)
        self.transformer = XLNetModel(config)
        self.classifier = XLNetClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_mems_train=False,
        use_mems_eval=False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict=False,
    ):
        r"""
        The XLNetForMultipleChoice forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                See :class:`XLNetModel`.
            token_type_ids (Tensor, optional):
                See :class:`XLNetModel`.
            attention_mask (Tensor, optional):
                See :class:`XLNetModel`.
            mems (Tensor, optional):
                See :class:`XLNetModel`.
            perm_mask (Tensor, optional):
                See :class:`XLNetModel`.
            target_mapping (Tensor, optional):
                See :class:`XLNetModel`.
            input_mask (Tensor, optional):
                See :class:`XLNetModel`.
            head_mask (Tensor, optional):
                See :class:`XLNetModel`.
            inputs_embeds (Tensor, optional):
                See :class:`XLNetModel`.
            use_mems_train (bool, optional):
                See :class:`XLNetModel`.
            use_mems_eval (bool, optional):
                See :class:`XLNetModel`.
            return_dict (bool, optional):
                See :class:`XLNetModel`.

        Returns:
            tensor or dict: Returns tensor `logtis` or a dict with key-value pairs:
             {"logits": `logits`, "mems": `mems`,
            "hidden_states": `hidden_states`, "attentions": `attentions`}

            With the corresponding fields:
            - `logits` (Tensor):
                Classification scores before SoftMax (also called logits). It's data type should be `float32`
                and has a shape of [batch_size, sequence_length, num_classes].
            - `mems` (List[Tensor]):
                See :class:`XLNetModel`.
            - `hidden_states` (List[Tensor], optional):
                See :class:`XLNetModel`.
            - `attentions` (List[Tensor], optional):
                See :class:`XLNetModel`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import XLNetForMultipleChoice, XLNetTokenizer
                from paddlenlp.data import Pad, Dict
                tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
                model = XLNetForMultipleChoice.from_pretrained('xlnet-base-cased')
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
        num_choices = paddle.shape(input_ids)[1] if input_ids is not None else paddle.shape(inputs_embeds)[1]
        input_ids = input_ids.reshape(shape=(-1, paddle.shape(input_ids)[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(shape=(-1, paddle.shape(attention_mask)[-1]))

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(shape=(-1, paddle.shape(token_type_ids)[-1]))

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.reshape(
                shape=(paddle.shape(inputs_embeds)[0], -1, paddle.shape(inputs_embeds)[-1])
            )

        transformer_outputs = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            input_mask=input_mask,
            head_mask=head_mask,
            use_mems_train=use_mems_train,
            use_mems_eval=use_mems_eval,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]
        logits = self.classifier(output)
        reshaped_logits = logits.reshape([-1, num_choices])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.reshape(shape=[-1]))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return tuple_output(output, loss)

        return XLNetForMultipleChoiceOutput(
            loss=loss,
            logits=reshaped_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLNetForQuestionAnswering(XLNetPretrainedModel):
    """
      XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
      layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

    Args:
        xlnet (:class:`XLNetModel`):
            An instance of :class:`XLNetModel`.
    """

    def __init__(self, config: XLNetConfig):
        super(XLNetForQuestionAnswering, self).__init__(config)
        self.transformer = XLNetModel(config)
        self.qa_outputs = nn.Linear(self.transformer.d_model, 2)

        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        start_positions=None,
        end_positions=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_mems_train=False,
        use_mems_eval=False,
        return_dict=False,
    ):
        r"""
        The XLNetForQuestionAnswering forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                See :class:`XLNetModel`.
            token_type_ids (Tensor, optional):
                See :class:`XLNetModel`.
            attention_mask (Tensor, optional):
                See :class:`XLNetModel`.
            mems (Tensor, optional):
                See :class:`XLNetModel`.
            perm_mask (Tensor, optional):
                See :class:`XLNetModel`.
            target_mapping (Tensor, optional):
                See :class:`XLNetModel`.
            input_mask (Tensor, optional):
                See :class:`XLNetModel`.
            head_mask (Tensor, optional):
                See :class:`XLNetModel`.
            inputs_embeds (Tensor, optional):
                See :class:`XLNetModel`.
            use_mems_train (bool, optional):
                See :class:`XLNetModel`.
            use_mems_eval (bool, optional):
                See :class:`XLNetModel`.
            return_dict (bool, optional):
                See :class:`XLNetModel`.

        Returns:
            tuple or dict: Returns tensor (`start_logits`, `end_logits`) or a dict with key-value pairs:
             {"start_logits": `start_logits`, "end_logits": `end_logits`, "mems": `mems`,
            "hidden_states": `hidden_states`, "attentions": `attentions`}

            With the corresponding fields:
            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].
            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].
            - `mems` (List[Tensor]):
                See :class:`XLNetModel`.
            - `hidden_states` (List[Tensor], optional):
                See :class:`XLNetModel`.
            - `attentions` (List[Tensor], optional):
                See :class:`XLNetModel`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.xlnet.modeling import XLNetForQuestionAnswering
                from paddlenlp.transformers.xlnet.tokenizer import XLNetTokenizer

                tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
                model = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)
                start_logits = outputs[0]
                end_logits = outputs[1]
        """
        transformer_outputs = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems_train=use_mems_train,
            use_mems_eval=use_mems_eval,
            return_dict=return_dict,
        )
        output = transformer_outputs[0]

        logits = self.qa_outputs(output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if start_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = paddle.shape(start_logits)[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = paddle.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            # the length of output must be larger than 1
            return tuple_output(output, loss)

        return XLNetForQuestionAnsweringSimpleOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


XLNetForCausalLM = XLNetLMHeadModel
