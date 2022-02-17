# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import paddle
from paddle import nn
from paddlenlp.transformers import PretrainedModel, register_base_model
import paddle.nn.functional as F
from ...ops import einsum

__all__ = [
    'MegatronBertModel', 'MegatronBertPretrainedModel',
    'MegatronBertForQuestionAnswering', 'MegatronBertForSequenceClassification',
    'MegatronBertForNextSentencePrediction', 'MegatronBertForCausalLM',
    'MegatronBertForPreTraining', 'MegatronBertForMaskedLM',
    'MegatronBertForMultipleChoice', 'MegatronBertForTokenClassification'
]


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


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + paddle.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_new": gelu_new,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "mish": mish,
    "linear": linear_act,
    "swish": swish,
}


class MegatronBertPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained MegatronBert models. It provides RoBerta related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "megatronbert-cased": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 2,
            "vocab_size": 29056,
            "pad_token_id": 0
        },
        "megatronbert-uncased": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 2,
            "vocab_size": 29056,
            "pad_token_id": 0
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "megatronbert-cased":
            "http://bj.bcebos.com/paddlenlp/models/transformers/"
            "megatron-bert/megatronbert-cased/model_state.pdparams",
            "megatronbert-uncased":
            "http://bj.bcebos.com/paddlenlp/models/transformers/"
            "megatron-bert/megatronbert-cased/model_state.pdparams",
        }
    }
    base_model_prefix = "megatronbert"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.megatronbert.config["initializer_range"],
                    shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = self.layer_norm_eps if hasattr(
                self, "layer_norm_eps") else self.megatronbert.config[
                    "layer_norm_eps"]


class MegatronBertEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self,
                 vocab_size=29056,
                 hidden_size=1024,
                 pad_token_id=0,
                 type_vocab_size=2,
                 max_position_embeddings=512,
                 hidden_dropout_prob=0.1,
                 position_embedding_type="absolute"):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.register_buffer(
            "position_ids",
            paddle.arange(end=max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = position_embedding_type

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:
                                             seq_length +
                                             past_key_values_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype='int64')

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings


class MegatronBertSelfAttention(nn.Layer):
    def __init__(self,
                 hidden_size=1024,
                 num_attention_heads=16,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 position_embedding_type=None):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads, self.attention_head_size
        ]
        x = x.reshape(new_x_shape)
        return x.transpose((0, 2, 1, 3))

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = paddle.matmul(query_layer,
                                         key_layer.transpose((0, 1, 3, 2)))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.shape[1]
            position_ids_l = paddle.arange(
                end=seq_length, dtype='int64').reshape((-1, 1))
            position_ids_r = paddle.arange(
                end=seq_length, dtype='int64').reshape((1, -1))
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = einsum("bhld,lrd->bhlr", query_layer,
                                                  positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in MegatronBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose((0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.all_head_size
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer, attention_probs


class MegatronBertSelfOutput(nn.Layer):
    def __init__(
            self,
            hidden_size=1024,
            hidden_dropout_prob=0.1, ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, residual):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return residual + hidden_states


class MegatronBertAttention(nn.Layer):
    def __init__(self,
                 hidden_size=1024,
                 layer_norm_eps=1e-12,
                 num_attention_heads=16,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 position_embedding_type=None):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.self = MegatronBertSelfAttention(
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            position_embedding_type=position_embedding_type)
        self.output = MegatronBertSelfOutput(
            hidden_size=hidden_size, hidden_dropout_prob=hidden_dropout_prob)
        self.pruned_heads = set()

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        ln_outputs = self.ln(hidden_states)
        self_outputs = self.self(ln_outputs, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,
                   ) + self_outputs[1:]  # add attentions if we output them
        return outputs


class MegatronBertIntermediate(nn.Layer):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = get_activation(hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MegatronBertOutput(nn.Layer):
    def __init__(self,
                 intermediate_size,
                 hidden_dropout_prob=0.1,
                 hidden_size=1024):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return input_tensor + hidden_states


# Based on transformers.models.bert.modeling_bert.BertLayer. Added LayerNorm.
class MegatronBertLayer(nn.Layer):
    def __init__(self,
                 hidden_size=1024,
                 hidden_act="gelu",
                 layer_norm_eps=1e-12,
                 num_attention_heads=16,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 intermediate_size=4096,
                 position_embedding_type=None):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = MegatronBertAttention(
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            position_embedding_type=position_embedding_type)

        self.ln = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.intermediate = MegatronBertIntermediate(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act)
        self.output = MegatronBertOutput(
            intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_size=hidden_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask,
                                                head_mask)
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]

        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output, ) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        ln_output = self.ln(attention_output)
        intermediate_output = self.intermediate(ln_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MegatronBertEncoder(nn.Layer):
    def __init__(self,
                 hidden_size=1024,
                 hidden_act="gelu",
                 layer_norm_eps=1e-12,
                 num_attention_heads=16,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 intermediate_size=4096,
                 position_embedding_type=None,
                 num_hidden_layers=24):
        super().__init__()
        self.layer = nn.LayerList([
            MegatronBertLayer(
                hidden_size=hidden_size,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
                num_attention_heads=num_attention_heads,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                max_position_embeddings=max_position_embeddings,
                intermediate_size=intermediate_size,
                position_embedding_type=position_embedding_type)
            for _ in range(num_hidden_layers)
        ])

        # The final layer norm. We removed the 1st LN, moved LN to each hidden layer and this one
        # is simply the final LN (Transformer's BERT has it attached to each hidden layer).
        self.ln = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, attention_mask,
                                         layer_head_mask)

            hidden_states = layer_outputs[0]

        # Finalize the hidden states.
        hidden_states = self.ln(hidden_states)

        return hidden_states


class MegatronBertPooler(nn.Layer):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@register_base_model
class MegatronBertModel(MegatronBertPretrainedModel):
    def __init__(self,
                 vocab_size=29056,
                 hidden_size=1024,
                 pad_token_id=0,
                 layer_norm_eps=1e-12,
                 type_vocab_size=2,
                 hidden_act="gelu",
                 attention_probs_dropout_prob=0.1,
                 num_attention_heads=16,
                 max_position_embeddings=512,
                 hidden_dropout_prob=0.1,
                 intermediate_size=4096,
                 num_hidden_layers=24,
                 initializer_range=0.02,
                 position_embedding_type="absolute"):
        super().__init__()

        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embeddings = MegatronBertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            pad_token_id=pad_token_id,
            type_vocab_size=type_vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=hidden_dropout_prob,
            position_embedding_type=position_embedding_type)
        self.encoder = MegatronBertEncoder(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            intermediate_size=intermediate_size,
            position_embedding_type=position_embedding_type,
            num_hidden_layers=num_hidden_layers)

        self.pooler = MegatronBertPooler(hidden_size=hidden_size)

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None):

        input_shape = input_ids.shape

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = paddle.ones(((batch_size, seq_length)))
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype='int64')

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or "
                f"attention_mask (shape {attention_mask.shape})")
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (:obj:`paddle.Tensor`): An attention mask.

        Returns:
            :obj:`paddle.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.ndim == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:,
                                                                     None, :, :]
        if encoder_attention_mask.ndim == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None,
                                                                     None, :]

        if encoder_extended_attention_mask.dtype == paddle.float16:
            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask) * -1e4
        elif encoder_extended_attention_mask.dtype == paddle.float32:
            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask) * -1e9

        return encoder_extended_attention_mask

    def get_head_mask(self,
                      head_mask,
                      num_hidden_layers,
                      is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask,
                                                      num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.ndim == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
                -1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.ndim == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                -1)  # We can specify head_mask for each layer
        assert head_mask.ndim == 5, f"head_mask.dim != 5, instead {len(head_mask.shape)}"
        return head_mask


class MegatronBertForQuestionAnswering(MegatronBertPretrainedModel):
    def __init__(self, megatronbert):
        super().__init__()
        self.megatronbert = megatronbert
        self.qa_outputs = nn.Linear(self.megatronbert.config['hidden_size'], 2)

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None):
        outputs = self.megatronbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(2, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        output = (start_logits, end_logits)
        return output


class MegatronBertForSequenceClassification(MegatronBertPretrainedModel):
    def __init__(self, megatronbert, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.megatronbert = megatronbert
        self.dropout = nn.Dropout(self.megatronbert.config[
            'hidden_dropout_prob'])
        self.classifier = nn.Linear(self.megatronbert.config['hidden_size'],
                                    num_labels)

        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None):
        outputs = self.megatronbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class MegatronBertPredictionHeadTransform(nn.Layer):
    def __init__(self, hidden_size, layer_norm_eps, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = get_activation(hidden_act)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MegatronBertLMPredictionHead(nn.Layer):
    def __init__(self, hidden_size, layer_norm_eps, vocab_size, hidden_act):
        super().__init__()
        self.transform = MegatronBertPredictionHeadTransform(
            hidden_size, layer_norm_eps, hidden_act)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MegatronBertOnlyMLMHead(nn.Layer):
    def __init__(self, hidden_size, layer_norm_eps, vocab_size, hidden_act):
        super().__init__()
        self.predictions = MegatronBertLMPredictionHead(
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            vocab_size=vocab_size,
            hidden_act=hidden_act)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class MegatronBertOnlyNSPHead(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class MegatronBertPreTrainingHeads(nn.Layer):
    def __init__(self, hidden_size, layer_norm_eps, vocab_size, hidden_act):
        super().__init__()
        self.predictions = MegatronBertLMPredictionHead(
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            vocab_size=vocab_size,
            hidden_act=hidden_act)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MegatronBertForPreTraining(MegatronBertPretrainedModel):
    def __init__(self, megatronbert):
        super().__init__()

        self.megatronbert = megatronbert
        self.cls = MegatronBertPreTrainingHeads(
            hidden_size=self.megatronbert.config['hidden_size'],
            layer_norm_eps=self.megatronbert.config['layer_norm_eps'],
            vocab_size=self.megatronbert.config['vocab_size'])

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None):
        outputs = self.megatronbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output,
                                                             pooled_output)

        output = (prediction_scores, seq_relationship_score)
        return output


class MegatronBertForCausalLM(MegatronBertPretrainedModel):
    def __init__(self, megatronbert):
        super().__init__()

        self.megatronbert = megatronbert
        self.cls = MegatronBertOnlyMLMHead(
            hidden_size=self.megatronbert.config['hidden_size'],
            layer_norm_eps=self.megatronbert.config['layer_norm_eps'],
            vocab_size=self.megatronbert.config['vocab_size'],
            hidden_act=self.megatronbert.config['hidden_act'])

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None):
        outputs = self.megatronbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        return prediction_scores


class MegatronBertForMaskedLM(MegatronBertPretrainedModel):
    def __init__(self, megatronbert):
        super().__init__()

        self.megatronbert = megatronbert
        self.cls = MegatronBertOnlyMLMHead(
            hidden_size=self.megatronbert.config['hidden_size'],
            layer_norm_eps=self.megatronbert.config['layer_norm_eps'],
            vocab_size=self.megatronbert.config['vocab_size'],
            hidden_act=self.megatronbert.config['hidden_act'])

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None, ):

        outputs = self.megatronbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        return prediction_scores


class MegatronBertForNextSentencePrediction(MegatronBertPretrainedModel):
    def __init__(self, megatronbert):
        super().__init__()

        self.megatronbert = megatronbert
        self.cls = MegatronBertOnlyNSPHead(
            hidden_size=self.megatronbert.config['hidden_size'])

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None):

        outputs = self.megatronbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask)

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        return seq_relationship_scores


class MegatronBertForMultipleChoice(MegatronBertPretrainedModel):
    def __init__(self, megatronbert):
        super().__init__()

        self.megatronbert = megatronbert
        self.dropout = nn.Dropout(self.megatronbert.config[
            'hidden_dropout_prob'])
        self.classifier = nn.Linear(self.megatronbert.config['hidden_size'], 1)

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.reshape(
            (-1, input_ids.shape[-1])) if input_ids is not None else None
        attention_mask = attention_mask.reshape(
            (-1,
             attention_mask.shape[-1])) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(
            (-1,
             token_type_ids.shape[-1])) if token_type_ids is not None else None
        position_ids = position_ids.reshape(
            (-1, position_ids.shape[-1])) if position_ids is not None else None

        outputs = self.megatronbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape((-1, num_choices))

        return reshaped_logits


class MegatronBertForTokenClassification(MegatronBertPretrainedModel):
    def __init__(self, megatronbert, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.megatronbert = megatronbert
        self.dropout = nn.Dropout(self.megatronbert.config[
            'hidden_dropout_prob'])
        self.classifier = nn.Linear(self.megatronbert.config['hidden_size'],
                                    self.num_labels)
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None):

        outputs = self.megatronbert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits
