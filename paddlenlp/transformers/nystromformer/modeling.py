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

import math
import re
from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
from matplotlib.transforms import TransformedPatchPath
from paddle import Tensor
from paddle.distributed.fleet.utils import recompute

from paddlenlp.utils.log import logger

from ...utils.env import CONFIG_NAME
from .. import PretrainedModel, register_base_model
from ..activations import ACT2FN
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from .configuration import (
    NYSTROMFORMER_PRETRAINED_INIT_CONFIGURATION,
    NYSTROMFORMER_PRETRAINED_RESOURCE_FILES_MAP,
    NystromformerConfig,
)
from .utils import (
    apply_chunking_to_forward,
    get_activation,
    get_extended_attention_mask,
    get_head_mask,
    trans_matrix,
)

__all__ = [
    "NystromformerEmbeddings",
    "NystromformerModel",
    "NystromformerPretrainedModel",
    "NystromformerForSequenceClassification",
    # "NystromformerForMaskedLM",
    # "NystromformerForTokenClassification",
    # "NystromformerForMultipleChoice",
    # "NystromformerForQuestionAnswering"
]


class NystromformerEmbeddings(nn.Layer):
    """
    Construct the embeddings from word, position and token_type embeddings
    """

    def __init__(self, config: NystromformerConfig):
        super(NystromformerEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", paddle.arange(config.max_position_embeddings).expand((1, -1)) + 2)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "token_type_ids",
            paddle.zeros(self.position_ids.shape, dtype=paddle.int64),
            persistable=False,
        )

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ):

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # Note: 确定下position_ids是否为[1, seq_length]
            position_ids = self.position_ids[:, :seq_length]
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids,
        # sloves the issue: https://github.com/huggingface/transformers/issues/5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand((input_shape[0], seq_length))
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

    def __init__(self, config: NystromformerConfig, position_embedding_type: Optional[str] = None):
        super(NystromformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
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
            config, "position_embedding_type", "absolute"
        )

        if self.conv_kernel_size is not None:
            self.conv = nn.Conv2D(
                in_channels=self.num_attention_heads,
                out_channels=self.num_attention_heads,
                kernel_size=(self.conv_kernel_size, 1),
                padding=(self.conv_kernel_size // 2, 0),
                bias_attr=False,
                groups=self.num_attention_heads,
            )

    # Function to approximate Moore-Penrose inverse via the iterative method
    def iterative_inv(self, mat, n_iter=6):
        identity = paddle.eye(mat.shape[-1])
        key = mat

        # The entries of key are positive and ||key||_{\infty} = 1 due to softmax
        if self.init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0.
            # TODO Fix trans_matrix
            value = 1 / paddle.max(paddle.sum(key, axis=-2)) * trans_matrix(key)
        else:
            # TODO make sure this way is OK
            # This is the exact coefficient computation, 1 / ||key||_1, of initialization of Z_0, leading to faster convergence.
            value = 1 / paddle.max(paddle.sum(key, axis=-2), axis=-1).values[:, :, None, None] * trans_matrix(key)

        for _ in range(n_iter):
            key_value = paddle.matmul(key, value)
            value = paddle.matmul(
                0.25 * value,
                13 * identity
                - paddle.matmul(key_value, 15 * identity - paddle.matmul(key_value, 7 * identity - key_value)),
            )
        return value

    def transpose_for_scores(self, layer):
        new_layer_shape = layer.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        layer = layer.reshape(new_layer_shape)
        return layer.transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[Tensor] = False,
    ):
        print("NystromformerAttention:-1")
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        print("NystromformerAttention:-2")

        # Note
        query_layer = query_layer / math.sqrt(math.sqrt(self.attention_head_size))
        key_layer = key_layer / math.sqrt(math.sqrt(self.attention_head_size))
        print("NystromformerAttention:-3")

        if self.num_landmarks == self.seq_len:
            print(self.num_landmarks, self.seq_len, "*")
            # [32, 12, 4096, 64] query_layer
            print(query_layer.shape, key_layer.shape, key_layer.transpose([0, 1, 3, 2]).shape)
            print(query_layer.mean(), key_layer.mean())
            attention_scores = paddle.matmul(query_layer, key_layer, transpose_y=True)
            # attention_scores = paddle.matmul(query_layer, key_layer.transpose([0,1,3,2]))
            # attention_scores = paddle.matmul(query_layer, trans_matrix(key_layer))
            print("3-1")
            # Note: make sure attention mask shape
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in NystromformerModel forward() function)
                attention_scores = attention_scores + attention_mask

            attention_probs = nn.functional.softmax(attention_scores, axis=-1)
            context_layer = paddle.matmul(attention_probs, value_layer)
        else:
            # Note: query_layer'shape: batch_size x num_heads x seq_len x head_size，
            # 进一步分析self.seq_len
            print("NystromformerAttention:-4")
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
            print("NystromformerAttention:-5")
            kernel_1 = nn.functional.softmax(paddle.matmul(query_layer, trans_matrix(k_landmarks)), axis=-1)
            kernel_2 = nn.functional.softmax(paddle.matmul(q_landmarks, trans_matrix(k_landmarks)), axis=-1)
            print("NystromformerAttention:-6")

            attention_scores = paddle.matmul(q_landmarks, trans_matrix(key_layer))
            print("NystromformerAttention:-7")

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in NystromformerModel forward() function)
                attention_scores = attention_scores + attention_mask

            print("NystromformerAttention:-8")
            kernel_3 = nn.functional.softmax(attention_scores, axis=-1)
            attention_probs = paddle.matmul(kernel_1, self.iterative_inv(kernel_2))
            print("NystromformerAttention:-9")
            new_value_layer = paddle.matmul(kernel_3, value_layer)
            context_layer = paddle.matmul(attention_probs, new_value_layer)
            print("NystromformerAttention:-10")

        if self.conv_kernel_size is not None:
            context_layer += self.conv(value_layer)

        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        # Note: batch_size x seq_length x hidden_size
        return outputs


class NystromformerSelfOutput(nn.Layer):
    """
    Output layer after a nystrom-based self-attention
    """

    def __init__(self, config: NystromformerConfig):
        super(NystromformerSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NystromformerAttention(nn.Layer):
    """
    A layer containing a nystromformer self-attention layer and its following output layer
    """

    def __init__(self, config: NystromformerConfig, position_embedding_type: Optional[str] = None):
        super(NystromformerAttention, self).__init__()
        self.self = NystromformerSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = NystromformerSelfOutput(config)
        self.pruned_heads = set()

    # TODO
    # def prune_heads(self, heads)

    def forward(
        self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None, output_attentions: Optional[bool] = False
    ):
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class NystromformerIntermediate(nn.Layer):
    """
    Nystromformer activation layer
    """

    def __init__(self, config):
        super(NystromformerIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class NystromformerOutput(nn.Layer):
    """
    Output layer after a nystromformer attention layer and its activation layer
    """

    def __init__(self, config: NystromformerConfig):
        super(NystromformerOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NystromformerLayer(nn.Layer):
    """
    Nystromformer layer containing attention, activation and output layer
    """

    def __init__(self, config: NystromformerConfig):
        super(NystromformerLayer, self).__init__()
        # Note
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = NystromformerAttention(config)
        # Note
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = NystromformerIntermediate(config)
        self.output = NystromformerOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[Tensor] = False,
    ):
        print("NystromformerLayer--")
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        print("NystromformerLayer--2")
        # Note: batch_size x seq_len x hiddensize
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        print("NystromformerLayer--3")
        # Note: apply_chunking_to_forward, 为减小内存占用，设置的chunk计算方式
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        print("NystromformerLayer--4")
        outputs = (layer_output,) + outputs
        print("NystromformerLayer--5")
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class NystromformerEncoder(nn.Layer):
    """
    Nystromformer encoder containing several nystromformer layers
    """

    def __init__(self, config: NystromformerConfig):
        super(NystromformerEncoder, self).__init__()
        self.config = config
        self.layer = nn.LayerList([NystromformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        print("input: ", hidden_states)
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            print(i, "***", self.gradient_checkpointing, self.training)
            if self.gradient_checkpointing and self.training:

                def create_cumtom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = recompute(create_cumtom_forward(layer_module), hidden_states, attention_mask)
            else:
                print("----")
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)
                print("---*-")

            hidden_states = layer_outputs[0]
            print("layer ", i + 1, ": ", hidden_states)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class NystromformerPredictionHeadTransform(nn.Layer):
    def __init__(self, config: NystromformerConfig):
        super(NystromformerPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class NystromformerLMPredictionHead(nn.Layer):
    def __init__(self, config: NystromformerConfig):
        super(NystromformerLMPredictionHead, self).__init__()
        self.transform = NystromformerPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        # Note
        self.bias = paddle.create_parameter(shape=(config.vocab_size,), dtype=self.decoder.weight.dtype)

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class NystromformerOnlyMLMHead(nn.Layer):
    def __init__(self, config: NystromformerConfig):
        super(NystromformerOnlyMLMHead, self).__init__()
        self.predictions = NystromformerLMPredictionHead(config)

    def forward(self, sequence_output: Tensor):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class NystromformerPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained Nystromformer models. It provides Nystromformer related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = CONFIG_NAME
    config_class = NystromformerConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "nystromformer"

    # model init configuration
    pretrained_init_configuration = NYSTROMFORMER_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = NYSTROMFORMER_PRETRAINED_RESOURCE_FILES_MAP

    # TODO 涉及到pruned_heads
    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
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

    def __init__(self, config: NystromformerConfig):
        super(NystromformerModel, self).__init__(config)
        self.config = config
        self.embeddings = NystromformerEmbeddings(config)
        self.encoder = NystromformerEncoder(config)
        # TODO 等待对齐
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # TODO 等待对齐
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], BaseModelOutputWithPastAndCrossAttentions]:
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
        # TODO 等待检查
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = paddle.ones((batch_size, seq_length))

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand((batch_size, seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # TODO 类型等待和attetion mask对齐
                token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # Note 缩减版get_extended_attention_mask
        extended_attention_mask: Tensor = get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # Note没有用到
        head_mask = get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # print("embedding_output: ", embedding_output.shape, embedding_output)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class NystromformerForMaskedLM(NystromformerPretrainedModel):
    """
    Nystromformer Model with a `masked language modeling` head on top.
    Args:
        config (:class:`ErnieConfig`):
            An instance of ErnieConfig used to construct ErnieForMaskedLM.
    """

    # Note 确定是否有用
    _keys_to_ignore_on_load_missing = ["cls.predictions.decoder"]

    def __init__(self, config: NystromformerConfig):
        super(NystromformerForMaskedLM, self).__init__(config)

        self.config = config
        self.nystromformer = NystromformerModel(config)
        self.cls = NystromformerOnlyMLMHead(config)

        self.apply(self.init_weights)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], MaskedLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.nystromformer(
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

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.reshpae([-1, self.config.vocab_size]), labels.reshpae([-1]))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NystromformerClassificationHead(nn.Layer):
    """
    Classification head of nystromformer used in sequence classification
    """

    def __init__(self, config: NystromformerConfig):
        super(NystromformerClassificationHead, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        # self.activation = get_activation(config.hidden_act)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
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

    def __init__(self, config: NystromformerConfig):
        super(NystromformerForSequenceClassification, self).__init__()
        self.num_labels = config.num_labels
        self.nystromformer = NystromformerModel(config)
        self.classifier = NystromformerClassificationHead(config)
        self.config = config if config is not None else NystromformerConfig()

        self.apply(self.init_weights)

    # def from_pretrained(self, pretrained_model_name_or_path, *args, **kwargs):
    #     self.nystromformer = \
    #         NystromformerModel(self.config).from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    #     print(
    #         "NystromformerModel is loaded with pretrained parameters, "
    #         "while the classification head is randomly initialized. "
    #         "You may need to fine-tune this model to achieve classification accuracy."
    #     )

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], SequenceClassifierOutput]:
        """
        # TODO 等待对齐参数
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.nystromformer(
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
        # Note 验证对齐
        # sequence_output = outputs['last_hidden_state']
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.flatten())
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NystromformerForMultipleChoice(NystromformerPretrainedModel):
    def __init__(self, config: NystromformerConfig):
        super(NystromformerForMultipleChoice, self).__init__(config)

        self.config = config
        self.nystromformer = NystromformerModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], MultipleChoiceModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Note: input_ids: batch_size x num_choice x seq_length
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.reshape([-1, input_ids.shape[-1]]) if input_ids is not None else None
        attention_mask = attention_mask.reshape([-1, attention_mask.shape[-1]]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape([-1, token_type_ids.shape[-1]]) if token_type_ids is not None else None
        position_ids = position_ids.reshape([-1, position_ids.shape[-1]]) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.reshape([-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1]])
            if inputs_embeds is not None
            else None
        )

        outputs = self.nystromformer(
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

        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.reshape([-1, num_choices])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NystromformerForTokenClassification(NystromformerPretrainedModel):
    def __init__(self, config: NystromformerConfig):
        super(NystromformerForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.nystromformer = NystromformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.nystromformer(
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

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.reshape([-1]))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NystromformerForQuestionAnswering(NystromformerPretrainedModel):
    def __init__(self, config: NystromformerConfig):
        super(NystromformerForQuestionAnswering, self).__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.nystromformer = NystromformerModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], QuestionAnsweringModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.nystromformer(
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

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                # Note check start_positions shape
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
