# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from .config import NystromformerConfig
from .. import PretrainedModel, register_base_model
from .utils import trans_matrix, get_activation, apply_chunking_to_forward, get_extended_attention_mask

__all__ = [
    "NystromformerEmbeddings", "NystromformerSelfAttention",
    "NystromformerSelfOutput", "NystromformerAttention",
    "NystromformerIntermediate", "NystromformerOutput", "NystromformerLayer",
    "NystromformerEncoder", "NystromformerPretrainedModel",
    "NystromformerModel", "NystromformerClassificationHead",
    "NystromformerForSequenceClassification"
]


class NystromformerEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings + 2, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            paddle.arange(config.max_position_embeddings).expand((1, -1)) + 2)
        self.position_embedding_type = getattr(config,
                                               "position_embedding_type",
                                               "absolute")
        self.register_buffer(
            "token_type_ids",
            paddle.zeros(self.position_ids.shape, dtype=paddle.int64),
            persistable=False,
        )

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NystromformerSelfAttention(nn.Layer):
    """
    Nystrom-based approximation of self-attention
    """
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})")
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_landmarks = config.num_landmarks
        self.seq_len = config.segment_means_seq_len
        self.conv_kernel_size = config.conv_kernel_size
        if config.inv_coeff_init_option:
            self.init_option = config["inv_init_coeff_option"]
        else:
            self.init_option = "original"
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute")
        if self.conv_kernel_size is not None:
            self.conv = nn.Conv2D(
                in_channels=self.num_attention_heads,
                out_channels=self.num_attention_heads,
                kernel_size=(self.conv_kernel_size, 1),
                padding=(self.conv_kernel_size // 2, 0),
                bias_attr=False,
                groups=self.num_attention_heads,
            )

    def iterative_inv(self, mat, n_iter=6):
        identity = paddle.eye(mat.size(-1))
        key = mat
        if self.init_option == "original":
            value = 1 / paddle.max(paddle.sum(key, axis=-2)) * trans_matrix(key)
        else:
            value = 1 / paddle.max(paddle.sum(key, axis=-2),
                                   axis=-1).values[:, :, None,
                                                   None] * trans_matrix(key)
        for _ in range(n_iter):
            key_value = paddle.matmul(key, value)
            value = paddle.matmul(
                0.25 * value,
                13 * identity - paddle.matmul(
                    key_value, 15 * identity -
                    paddle.matmul(key_value, 7 * identity - key_value)),
            )
        return value

    def transpose_for_scores(self, layer):
        new_layer_shape = layer.shape[:-1] + [
            self.num_attention_heads, self.attention_head_size
        ]
        layer = layer.reshape(new_layer_shape)
        return layer.transpose([0, 2, 1, 3])

    def forward(self,
                hidden_states,
                attention_mask=None,
                output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer = query_layer / paddle.sqrt(
            paddle.sqrt(self.attention_head_size))
        key_layer = key_layer / paddle.sqrt(
            paddle.sqrt(self.attention_head_size))
        if self.num_landmarks == self.seq_len:
            attention_scores = paddle.matmul(query_layer,
                                             trans_matrix(key_layer))
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = nn.functional.softmax(attention_scores, axis=-1)
            context_layer = paddle.matmul(attention_probs, value_layer)
        else:
            q_landmarks = query_layer.reshape(
                -1,
                self.num_attention_heads,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.attention_head_size,
            ).mean(axis=-2)
            k_landmarks = key_layer.reshape(
                -1,
                self.num_attention_heads,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.attention_head_size,
            ).mean(axis=-2)
            kernel_1 = nn.functional.softmax(paddle.matmul(
                query_layer, trans_matrix(k_landmarks)),
                                             axis=-1)
            kernel_2 = nn.functional.softmax(paddle.matmul(
                q_landmarks, trans_matrix(k_landmarks)),
                                             axis=-1)
            attention_scores = paddle.matmul(q_landmarks,
                                             trans_matrix(key_layer))
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            kernel_3 = nn.functional.softmax(attention_scores, axis=-1)
            attention_probs = paddle.matmul(kernel_1,
                                            self.iterative_inv(kernel_2))
            new_value_layer = paddle.matmul(kernel_3, value_layer)
            context_layer = paddle.matmul(attention_probs, new_value_layer)

        if self.conv_kernel_size is not None:
            context_layer += self.conv(value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.all_head_size
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)
        outputs = (context_layer,
                   attention_probs) if output_attentions else (context_layer, )
        return outputs


class NystromformerSelfOutput(nn.Layer):
    """
    Output layer after a nystrom-based self-attention
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NystromformerAttention(nn.Layer):
    """
    A layer containing a nystromformer self-attention layer and its following output layer
    """
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = NystromformerSelfAttention(
            config, position_embedding_type=position_embedding_type)
        self.output = NystromformerSelfOutput(config)
        self.pruned_heads = set()

    def forward(self,
                hidden_states,
                attention_mask=None,
                output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask,
                                 output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,
                   ) + self_outputs[1:]  # add attentions if we output them
        return outputs


class NystromformerIntermediate(nn.Layer):
    """
    Nystromformer activation layer
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class NystromformerOutput(nn.Layer):
    """
    Output layer after a nystromformer attention layer and its activation layer
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NystromformerLayer(nn.Layer):
    """
    Nystromformer layer containing attention, activation and output layer
    """
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = NystromformerAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = NystromformerIntermediate(config)
        self.output = NystromformerOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None,
                output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk,
                                                 self.chunk_size_feed_forward,
                                                 self.seq_len_dim,
                                                 attention_output)
        outputs = (layer_output, ) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class NystromformerEncoder(nn.Layer):
    """
    Nystromformer encoder containing several nystromformer layers
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.LayerList([
            NystromformerLayer(config) for _ in range(config.num_hidden_layers)
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
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )
            layer_outputs = layer_module(hidden_states, attention_mask,
                                         output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1], )
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions
        }


class NystromformerPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained Nystromformer models. It provides Nystromformer related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    base_model_prefix = "nystromformer"
    model_config_file = "config.json"

    # model init configuration
    pretrained_init_configuration = {
        "nystromformer-512": {
            "model_type": "nystromformer",
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 0,
            "conv_kernel_size": 65,
            "eos_token_id": 2,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "inv_coeff_init_option": False,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 510,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "num_landmarks": 64,
            "pad_token_id": 1,
            "segment_means_seq_len": 64,
            "type_vocab_size": 2,
            "use_cache": True,
            "vocab_size": 30000
        }
    }
    resource_files_names = {"model_state": "nystromformer_model.params"}
    pretrained_resource_files_map = {
        "model_state": {
            "nystromformer-512":
            # TODO: upload parameter file and get the link here.
            "https://to-be-uploaded"
        }
    }

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError


@register_base_model
class NystromformerModel(NystromformerPretrainedModel):
    """
    The bare Nystromformer Model outputting raw hidden-states.
    Nystromformer is a nystrom-based approximation of transformer which reduce the time complexity to O(n).
    See the Nystromformer paper at: https://arxiv.org/pdf/2102.03902v3.pdf

    Ref:
        Xiong, Yunyang, et al. "Nyströmformer: A Nystöm-based Algorithm for Approximating Self-Attention." AAAI, 2021.

    Args:
        config(NystromformerConfig):
            Configuration of the Nystromformer model, including hidden_size, hidden_layer_num, et al.
            See docs in paddlenlp.transformers.nystromformer.NystromformerConfig for more details.
            Defaults to `None`, which means using default configuration.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else NystromformerConfig()
        self.embeddings = NystromformerEmbeddings(config)
        self.encoder = NystromformerEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None):
        """
        The NystromformerModel forward method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type should be int. The `masked` tokens have `0` values and the others have `1` values.
                It is a tensor with shape `[batch_size, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                ``[0, max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            inputs_embeds (Tensor, optional):
                Indices of embedded input sequence. They are representations of tokens that build the input sequence.
                Its data type should be `float32` and it has a shape of [batch_size, sequence_length, hidden_size].
                Defaults to 'None', which means the input_ids represents the sequence.
            output_attentions (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.

        Returns:
            dict: Returns dict with keys of ('last_hidden_state', 'hidden_states', 'attentions')

        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import NystromformerModel, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained('nystromformer-512')
                model = BertModel.from_pretrained('nystromformer-512')
                inputs = tokenizer("Welcome to Baidu")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")
        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = paddle.ones((batch_size, seq_length))
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :
                                                                         seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)
        extended_attention_mask: paddle.Tensor = get_extended_attention_mask(
            attention_mask, input_shape)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        return encoder_outputs


class NystromformerClassificationHead(nn.Layer):
    """
    Classification head of nystromformer used in sequence classification
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.activation = get_activation(config.hidden_act)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NystromformerForSequenceClassification(NystromformerPretrainedModel):
    """
    Nystromformer Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        config(NystromformerConfig, optional):
            Configuration of the Nystromformer model, including hidden_size, hidden_layer_num, et al.
            See docs in paddlenlp.transformers.nystromformer.NystromformerConfig for more details.
            Defaults to `None`, which means using default configuration.
    """
    def __init__(self, config=None):
        super().__init__()
        self.num_labels = config.num_labels
        self.nystromformer = NystromformerModel(config)
        self.classifier = NystromformerClassificationHead(config)
        self.config = config if config is not None else NystromformerConfig()

    def from_pretrained(self, pretrained_model_name_or_path, *args, **kwargs):
        self.nystromformer = \
            NystromformerModel(self.config).from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        print(
            "NystromformerModel is loaded with pretrained parameters, "
            "while the classification head is randomly initialized. "
            "You may need to fine-tune this model to achieve classification accuracy."
        )

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None):
        """
        The NystromformerForSequenceClassification forward method

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type should be int. The `masked` tokens have `0` values and the others have `1` values.
                It is a tensor with shape `[batch_size, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                ``[0, max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            inputs_embeds (Tensor, optional):
                Indices of embedded input sequence. They are representations of tokens that build the input sequence.
                Its data type should be `float32` and it has a shape of [batch_size, sequence_length, hidden_size].
                Defaults to 'None', which means the input_ids represents the sequence.
            labels (Tensor, optional):
                Labels of the inputs.
            output_attentions (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.

        Returns:
            Dict of loss, logits, hidden_states and attentions

        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers.nystromformer.modeling import NystromformerForSequenceClassification
                from paddlenlp.transformers import NystromformerModel, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained('nystromformer-512')
                model = BertForSequenceClassification.from_pretrained('nystromformer-512')
                inputs = tokenizer("Welcome to Baidu")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)
        """
        outputs = self.nystromformer(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     inputs_embeds=inputs_embeds,
                                     output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states)
        sequence_output = outputs['last_hidden_state']
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64
                                              or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape([-1, self.num_labels]),
                                labels.flatten())
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs['hidden_states'],
            'attentions': outputs['attentions']
        }
