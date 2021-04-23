# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
"""Modeling classes for ALBERT model."""

import math
import re
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Layer
from .. import PretrainedModel, register_base_model

__all__ = [
    "AlbertModel",
    "AlbertPretrainedModel",
    "AlbertForPreTraining",
    "AlbertForMaskedLM",
    "AlbertForSequenceClassification",
    "AlbertForTokenClassification",
    "AlbertForMultipleChoice",
]

dtype_float = paddle.get_default_dtype()


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
            activation_string, list(ACT2FN.keys())))


def mish(x):
    return x * F.tanh(F.softplus(x))


def linear_act(x):
    return x


def swish(x):
    return x * F.sigmoid(x)


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "mish": mish,
    "linear": linear_act,
    "swish": swish,
}


def einsum4x4(equation, x, y):
    """
    Only works for 4D x 4D.
    """
    idx_x, idx_y, idx_z = re.split(",|->", equation)
    # Compute repeated index
    repeated_idx = list(set(idx_x + idx_y) - set(idx_z))

    unique_idx_x = list(set(idx_x) - set(idx_y))
    unique_idx_y = list(set(idx_y) - set(idx_x))
    common_idx = list(set(idx_x) & set(idx_y) - set(repeated_idx))

    new_idx_x = common_idx + unique_idx_x + repeated_idx
    new_idx_y = common_idx + unique_idx_y + repeated_idx
    new_idx_z = common_idx + unique_idx_x + unique_idx_y

    perm_x = [idx_x.index(i) for i in new_idx_x]
    perm_y = [idx_y.index(i) for i in new_idx_y]
    perm_z = [new_idx_z.index(i) for i in idx_z]

    x = paddle.transpose(x, perm=perm_x)
    y = paddle.transpose(y, perm=perm_y)
    z = paddle.matmul(x=x, y=y, transpose_y=True)
    z = paddle.transpose(z, perm=perm_z)
    return z


class AlbertEmbeddings(Layer):
    """
    Constructs the embeddings from word, position and token_type embeddings.
    """

    def __init__(
            self,
            vocab_size,
            embedding_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
            position_embedding_type,
            pad_token_id,
    ):
        super(AlbertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embedding_size)

        self.layer_norm = nn.LayerNorm(embedding_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", paddle.arange(max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = position_embedding_type

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertAttention(Layer):
    def __init__(
            self,
            embedding_size,
            hidden_size,
            num_attention_heads,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            max_position_embeddings,
            layer_norm_eps,
            position_embedding_type,
    ):
        super(AlbertAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 and not embedding_size:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.pruned_heads = set()

        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * max_position_embeddings - 1, self.attention_head_size)

    # Copied from transformers.models.bert.modeling_bert.BertSelfAttention.transpose_for_scores
    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # todo

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_layer, key_layer, transpose_y=True)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.shape[1]
            position_ids_l = paddle.arange(seq_length, dtype="int64").reshape((-1, 1))
            position_ids_r = paddle.arange(seq_length, dtype="int64").reshape((1, -1))

            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = paddle.cast(positional_embedding, query_layer.dtype)  # fp16 compatibility

            positional_embedding = paddle.stack([positional_embedding] * query_layer.shape[0], axis=0)
            if self.position_embedding_type == "relative_key":
                relative_position_scores = einsum4x4("bhld,blrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = einsum4x4("bhld,blrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = einsum4x4("bhrd,blrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])

        context_layer = context_layer.reshape([0, 0, -1])

        # dense layer shape to be checked
        projected_context_layer = self.dense(context_layer)

        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layer_normed_context_layer = self.layer_norm(hidden_states + projected_context_layer_dropout)
        return (layer_normed_context_layer, attention_probs) if output_attentions else (layer_normed_context_layer,)


class AlbertLayer(Layer):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 num_attention_heads,
                 intermediate_size,
                 hidden_act,
                 hidden_dropout_prob,
                 attention_probs_dropout_prob,
                 max_position_embeddings,
                 layer_norm_eps,
                 position_embedding_type,
    ):
        super(AlbertLayer, self).__init__()
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.attention = AlbertAttention(
            embedding_size,
            hidden_size,
            num_attention_heads,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            max_position_embeddings,
            layer_norm_eps,
            position_embedding_type,
        )
        self.ffn = nn.Linear(hidden_size, intermediate_size)
        self.ffn_output = nn.Linear(intermediate_size, hidden_size)
        self.activation = ACT2FN[hidden_act]
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)

        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them


class AlbertLayerGroup(Layer):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 num_attention_heads,
                 intermediate_size,
                 inner_group_num,
                 hidden_act,
                 hidden_dropout_prob,
                 attention_probs_dropout_prob,
                 max_position_embeddings,
                 layer_norm_eps,
                 position_embedding_type,
    ):
        super(AlbertLayerGroup, self).__init__()

        self.albert_layers = nn.LayerList([
                AlbertLayer(
                    embedding_size,
                    hidden_size,
                    num_attention_heads,
                    intermediate_size,
                    hidden_act,
                    hidden_dropout_prob,
                    attention_probs_dropout_prob,
                    max_position_embeddings,
                    layer_norm_eps,
                    position_embedding_type,
                ) for _ in range(inner_group_num)
        ])

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
    ):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index], output_attentions)
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class AlbertTransformer(Layer):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 num_hidden_layers,
                 num_hidden_groups,
                 num_attention_heads,
                 intermediate_size,
                 inner_group_num,
                 hidden_dropout_prob,
                 attention_probs_dropout_prob,
                 max_position_embeddings,
                 layer_norm_eps,
                 position_embedding_type,
    ):
        super(AlbertTransformer, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups

        self.embedding_hidden_mapping_in = nn.Linear(embedding_size, hidden_size)
        self.albert_layer_groups = nn.LayerList([
            AlbertLayerGroup(
                embedding_size,
                hidden_size,
                num_attention_heads,
                intermediate_size,
                inner_group_num,
                hidden_act,
                hidden_dropout_prob,
                attention_probs_dropout_prob,
                max_position_embeddings,
                layer_norm_eps,
                position_embedding_type,
            ) for _ in range(num_hidden_groups)
        ])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i in range(self.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.num_hidden_layers / self.num_hidden_groups)
            # Index of the hidden group
            group_idx = int(i / (self.num_hidden_layers / self.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }


class AlbertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ALBERT models. It provides ALBERT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "albert-base-v1": {},
        "albert-large-v1": {},
        "albert-xlarge-v1": {},
        "albert-xxlarge-v1": {},
        "albert-base-v2": {
            "attention_probs_dropout_prob": 0,
            "bos_token_id": 2,
            "classifier_dropout_prob": 0.1,
            "down_scale_factor": 1,
            "embedding_size": 128,
            "eos_token_id": 3,
            "gap_size": 0,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "albert",
            "net_structure_type": 0,
            "num_attention_heads": 12,
            "num_hidden_groups": 1,
            "num_hidden_layers": 12,
            "num_memory_blocks": 0,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30000
        },
        "albert-large-v2": {},
        "albert-xlarge-v2": {},
        "albert-xxlarge-v2": {},
    }

    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "albert-base-v1": "",
            "albert-large-v1": "",
            "albert-xlarge-v1": "",
            "albert-xxlarge-v1": "",
            "albert-base-v2": "",
            "albert-large-v2": "",
            "albert-xlarge-v2": "",
            "albert-xxlarge-v2": "",
        }
    }
    base_model_prefix = "albert"

    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        # Initialize the weights.
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.transformer.config["initializer_range"],
                    shape=layer.weight.shape)
            )
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.transformer.config["initializer_range"],
                    shape=layer.weight.shape)
            )
            if layer._padding_idx is not None:
                layer.weight[layer._padding_idx].set_value(
                    paddle.zeros_like(layer.weight[layer._padding_idx])
                )
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.ones_like(layer.weight))


@register_base_model
class AlbertModel(AlbertPretrainedModel):
    def __init__(self,
                 vocab_size=30000,
                 embedding_size=128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_hidden_groups=1,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 inner_group_num=1,
                 hidden_act="gelu",
                 hidden_dropout_prob=0,
                 attention_probs_dropout_prob=0,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 classifier_dropout_prob=0.1,
                 position_embedding_type="absolute",
                 pad_token_id=0,
                 bos_token_id=2,
                 eos_token_id=3,
                 add_pooling_layer=True,
        ):
        super(AlbertModel, self).__init__()

        self.embeddings = AlbertEmbeddings(
            vocab_size,
            embedding_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
            position_embedding_type,
            pad_token_id,)

        self.encoder = AlbertTransformer(
            embedding_size,
            hidden_size,
            num_hidden_layers,
            num_hidden_groups,
            num_attention_heads,
            intermediate_size,
            inner_group_num,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            max_position_embeddings,
            layer_norm_eps,
            position_embedding_type,)

        if add_pooling_layer:
            self.pooler = nn.Linear(hidden_size, hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        pass

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            num_hidden_layers=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.place if input_ids is not None else inputs_embeds.place

        if attention_mask is None:
            attention_mask = paddle.ones(shape=input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(shape=input_shape, dtype="int64")

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = paddle.cast(extended_attention_mask, dtype=dtype_float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # todo: get head mask
        # head_mask = self.get_head_mask(head_mask, num_hidden_layers)
        head_mask = [None] * num_hidden_layers

        embedding_output = self.embeddings(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
            "hidden_states": encoder_outputs.hidden_states,
            "attentions": encoder_outputs.attentions,
        }


class AlbertForPretraining(AlbertPretrainedModel):
    def __init__(self, albert, lm_head, sop_head, vocab_size):
        super(AlbertForPreTraining, self).__init__()

        self.albert = albert
        self.predictions = lm_head
        self.sop_classifier = sop_head
        self.init_weights()
        self.vocab_size = vocab_size

    def get_output_embeddings(self):
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.predictions.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.albert.embeddings.word_embeddings

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                sentence_order_label=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
    ):
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]

        prediction_scores = self.predictions(sequence_output)
        sop_scores = self.sop_classifier(pooled_output)

        total_loss = None
        if labels is not None and sentence_order_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.reshape((-1, self.vocab_size)), labels.reshape((-1)))
            sentence_order_loss = loss_fct(sop_scores.reshape((-1, 2)), sentence_order_label.reshape((-1)))
            total_loss = masked_lm_loss + sentence_order_loss

        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return {
            "loss": total_loss,
            "prediction_logits": prediction_scores,
            "sop_logits": sop_scores,
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
        }


class AlbertMLMHead(Layer):
    def __init__(self,
                 embedding_size,
                 vocab_size,
                 hidden_size,
                 hidden_act,
    ):
        super(AlbertMLMHead, self).__init__()

        self.layer_norm = nn.LayerNorm(embedding_size)
        self.bias = self.create_parameter(
            [vocab_size],
            is_bias=True,
            default_initializer=nn.initializer.Constant(value=0)
        )
        self.dense = nn.Linear(hidden_size, embedding_size)
        self.decoder = nn.Linear(embedding_size, vocab_size)
        self.activation = ACT2FN[hidden_act]

        # link bias
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states
        print(prediction_scores)


class AlbertSOPHead(Layer):
    def __init__(self,
                 classifier_dropout_prob,
                 hidden_size,
                 num_labels,
    ):
        super(AlbertSOPHead, self).__init__()
        self.dropout = nn.Dropout(classifier_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output):
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


# To begin with
class AlbertForMaskedLM(AlbertPretrainedModel):
    pass


class AlbertForSequenceClassification(AlbertPretrainedModel):
    pass


class AlbertForTokenClassification(AlbertPretrainedModel):
    pass


class AlbertForMultipleChoice(AlbertPretrainedModel):
    pass
