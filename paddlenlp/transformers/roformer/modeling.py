# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import tensor
from paddle.fluid import layers
from paddle.nn import Layer
from paddle.nn.layer.transformer import (
    MultiHeadAttention,
    _convert_attention_mask,
    _convert_param_attr_to_list, )
from ...ops.einsum import einsum
from .. import PretrainedModel, register_base_model

__all__ = [
    "RoFormerModel",
    "RoFormerPretrainedModel",
    "RoFormerForPretraining",
    "RoFormerPretrainingCriterion",
    "RoFormerPretrainingHeads",
    "RoFormerForSequenceClassification",
    "RoFormerForTokenClassification",
    "RoFormerForQuestionAnswering",
]

dtype_float = paddle.get_default_dtype()


class MultiHeadAttentionNew(MultiHeadAttention):
    def positional_embedding(self, inputs):
        seq_len = inputs.shape[1]
        pos_seq = paddle.arange(0, seq_len, dtype=dtype_float)
        indices = paddle.arange(0, self.head_dim, 2, dtype=dtype_float)
        indices = 1 / 10000**(indices / self.head_dim)
        sinusoid_inp = einsum("i,d->id", pos_seq, indices)
        pos_emb = paddle.concat(
            [paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=-1)
        pos_emb = paddle.reshape(pos_emb, (1, 1, seq_len, self.head_dim))
        pos_emb.stop_gradient = True

        return pos_emb

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos,
                                         query_layer,
                                         key_layer,
                                         value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]

        sin, cos = paddle.chunk(sinusoidal_pos, 2, axis=-1)
        paddle.reshape
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = paddle.reshape(
            paddle.stack(
                [sin, sin], axis=-1), sinusoidal_pos.shape)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = paddle.reshape(
            paddle.stack(
                [cos, cos], axis=-1), sinusoidal_pos.shape)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]

        rotate_half_query_layer = paddle.reshape(
            paddle.stack(
                [-query_layer[:, :, :, 1::2], query_layer[:, :, :, ::2]],
                axis=-1),
            query_layer.shape, )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]

        rotate_half_key_layer = paddle.reshape(
            paddle.stack(
                [-key_layer[:, :, :, 1::2], key_layer[:, :, :, ::2]], axis=-1),
            key_layer.shape, )
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = paddle.reshape(
                paddle.stack(
                    [-value_layer[:, :, :, 1::2], value_layer[:, :, :, ::2]],
                    axis=-1),
                value_layer.shape, )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if cache is None:
            q, k, v = self._prepare_qkv(query, key, value, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, cache)

        sinusoidal_pos = self.positional_embedding(query)
        q, k = self.apply_rotary_position_embeddings(sinusoidal_pos, q, k)

        # scale dot product attention
        # TODO(guosheng): use tensor.matmul, however it doesn't support `alpha`
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)
        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerEncoderLayerNew(nn.TransformerEncoderLayer):
    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward,
            dropout=0.1,
            activation="relu",
            attn_dropout=None,
            act_dropout=None,
            normalize_before=False,
            weight_attr=None,
            bias_attr=None, ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout=dropout,
            activation=activation,
            attn_dropout=attn_dropout,
            act_dropout=act_dropout,
            normalize_before=normalize_before,
            weight_attr=weight_attr,
            bias_attr=bias_attr, )
        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)
        self.self_attn = MultiHeadAttentionNew(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0], )


class RoFormerEmbeddings(Layer):
    """
    Include embeddings from word and token_type embeddings
    """

    def __init__(
            self,
            vocab_size,
            hidden_size=768,
            hidden_dropout_prob=0.1,
            type_vocab_size=2, ):
        super(RoFormerEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RoFormerPooler(Layer):
    """ """

    def __init__(self, hidden_size, pool_act="tanh"):
        super(RoFormerPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class RoFormerPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained RoFormer models. It provides RoFormer related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "roformer-chinese-base": {
            "vocab_size": 50000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1536,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "roformer-chinese-char-base": {
            "vocab_size": 12000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "roformer-chinese-base":
            "https://huggingface.co/junnyu/roformer_chinese_base/resolve/main/model_state.pdparams",
            "roformer-chinese-char-base":
            "https://huggingface.co/junnyu/roformer_chinese_char_base/resolve/main/model_state.pdparams",
        }
    }
    base_model_prefix = "roformer"

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.roformer.config["initializer_range"],
                        shape=layer.weight.shape, ))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class RoFormerModel(RoFormerPretrainedModel):
    """
    The bare RoFormer Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Check the superclass documentation for the generic methods and the library implements for all its model.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (`int`):
            Vocabulary size of the XLNet model. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling XLNetModel.
        hidden_size (`int`, optional):
            Dimensionality of the encoder layers and the pooler layer. Defaults to ``768``.
        num_hidden_layers (`int`, optional):
            Number of hidden layers in the Transformer encoder. Defaults to ``12``.
        num_attention_heads (`int`, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to ``12``.
        intermediate_size (`int`, optional):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            Defaults to ``3072``.
        hidden_act (`str`, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to ``0.1``.
        attention_probs_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the pooler.
            Defaults to ``0.1``.
        initializer_range (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to ``0.02``.
    """

    def __init__(
            self,
            vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=1536,
            type_vocab_size=2,
            initializer_range=0.02,
            pad_token_id=0,
            pool_act="tanh", ):
        super(RoFormerModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = RoFormerEmbeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            type_vocab_size, )
        encoder_layer = TransformerEncoderLayerNew(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0, )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = RoFormerPooler(hidden_size, pool_act)
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None,
            output_hidden_states=False, ):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2], )
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids, )

        if output_hidden_states:
            output = embedding_output
            encoder_outputs = [embedding_output]
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            pooled_output = self.pooler(encoder_outputs[-1])
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)
            pooled_output = self.pooler(sequence_output)
        if output_hidden_states:
            return encoder_outputs, pooled_output
        else:
            return sequence_output, pooled_output


class RoFormerForQuestionAnswering(RoFormerPretrainedModel):
    def __init__(self, roformer, dropout=None):
        super(RoFormerForQuestionAnswering, self).__init__()
        self.roformer = roformer  # allow roformer to be config
        self.classifier = nn.Linear(self.roformer.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None):
        sequence_output, _ = self.roformer(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            attention_mask=None, )

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class RoFormerForSequenceClassification(RoFormerPretrainedModel):
    """
    Model for sentence (pair) classification task with RoFormer.
    Args:
        roformer (RoFormerModel): An instance of RoFormerModel.
        num_classes (int, optional): The number of classes. Default 2
        dropout (float, optional): The dropout probability for output of RoFormer.
            If None, use the same value as `hidden_dropout_prob` of `RoFormerModel`
            instance `roformer`. Default None
    """

    def __init__(self, roformer, num_classes=2, dropout=None):
        super(RoFormerForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.roformer = roformer  # allow roformer to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.roformer.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roformer.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, pooled_output = self.roformer(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask, )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RoFormerForTokenClassification(RoFormerPretrainedModel):
    def __init__(self, roformer, num_classes=2, dropout=None):
        super(RoFormerForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.roformer = roformer  # allow roformer to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.roformer.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roformer.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, _ = self.roformer(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask, )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class RoFormerLMPredictionHead(Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(RoFormerLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = (self.create_parameter(
            shape=[vocab_size, hidden_size],
            dtype=self.transform.weight.dtype,
            is_bias=False, ) if embedding_weights is None else
                               embedding_weights)
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = (paddle.tensor.matmul(
            hidden_states, self.decoder_weight, transpose_y=True) +
                         self.decoder_bias)
        return hidden_states


class RoFormerPretrainingHeads(Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(RoFormerPretrainingHeads, self).__init__()
        self.predictions = RoFormerLMPredictionHead(
            hidden_size, vocab_size, activation, embedding_weights)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class RoFormerForPretraining(RoFormerPretrainedModel):
    def __init__(self, roformer):
        super(RoFormerForPretraining, self).__init__()
        self.roformer = roformer
        self.cls = RoFormerPretrainingHeads(
            self.roformer.config["hidden_size"],
            self.roformer.config["vocab_size"],
            self.roformer.config["hidden_act"],
            embedding_weights=self.roformer.embeddings.word_embeddings.weight, )

        self.apply(self.init_weights)

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None,
            masked_positions=None, ):
        with paddle.static.amp.fp16_guard():
            outputs = self.roformer(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask, )
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_positions)
            return prediction_scores, seq_relationship_score


class RoFormerPretrainingCriterion(paddle.nn.Layer):
    def __init__(self, vocab_size):
        super(RoFormerPretrainingCriterion, self).__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(
            self,
            prediction_scores,
            seq_relationship_score,
            masked_lm_labels,
            next_sentence_labels,
            masked_lm_scale, ):
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = F.cross_entropy(
                prediction_scores,
                masked_lm_labels,
                reduction="none",
                ignore_index=-1)
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            next_sentence_loss = F.cross_entropy(
                seq_relationship_score, next_sentence_labels, reduction="none")
        return paddle.sum(masked_lm_loss) + paddle.mean(next_sentence_loss)
