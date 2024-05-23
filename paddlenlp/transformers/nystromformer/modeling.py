# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import math
from typing import Callable, Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.distributed.fleet.utils import recompute

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

__all__ = [
    "NystromformerEmbeddings",
    "NystromformerModel",
    "NystromformerPretrainedModel",
    "NystromformerForSequenceClassification",
    "NystromformerForMaskedLM",
    "NystromformerForTokenClassification",
    "NystromformerForMultipleChoice",
    "NystromformerForQuestionAnswering",
]


class NystromformerEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings.
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
        self.register_buffer(
            "position_ids", paddle.arange(config.max_position_embeddings, dtype="int64").expand((1, -1)) + 2
        )
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
            value = 1 / paddle.max(paddle.sum(key, axis=-2)) * key.transpose([0, 1, 3, 2])
        else:
            # TODO make sure this way is OK
            # This is the exact coefficient computation, 1 / ||key||_1, of initialization of Z_0, leading to faster convergence.
            value = (
                1
                / paddle.max(paddle.sum(key, axis=-2), axis=-1).values[:, :, None, None]
                * key.transpose([0, 1, 3, 2])
            )

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
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        query_layer = query_layer / math.sqrt(math.sqrt(self.attention_head_size))
        key_layer = key_layer / math.sqrt(math.sqrt(self.attention_head_size))

        if self.num_landmarks == self.seq_len:
            attention_scores = paddle.matmul(query_layer, key_layer, transpose_y=True)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in NystromformerModel forward() function)
                attention_scores = attention_scores + attention_mask

            attention_probs = nn.functional.softmax(attention_scores, axis=-1)
            context_layer = paddle.matmul(attention_probs, value_layer)
        else:
            q_landmarks = query_layer.reshape(
                [
                    -1,
                    self.num_attention_heads,
                    self.num_landmarks,
                    self.seq_len // self.num_landmarks,
                    self.attention_head_size,
                ]
            ).mean(axis=-2)
            k_landmarks = key_layer.reshape(
                [
                    -1,
                    self.num_attention_heads,
                    self.num_landmarks,
                    self.seq_len // self.num_landmarks,
                    self.attention_head_size,
                ]
            ).mean(axis=-2)

            kernel_1 = nn.functional.softmax(paddle.matmul(query_layer, k_landmarks, transpose_y=True), axis=-1)
            kernel_2 = nn.functional.softmax(paddle.matmul(q_landmarks, k_landmarks, transpose_y=True), axis=-1)

            attention_scores = paddle.matmul(q_landmarks, key_layer, transpose_y=True)

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in NystromformerModel forward() function)
                attention_scores = attention_scores + attention_mask

            kernel_3 = nn.functional.softmax(attention_scores, axis=-1)
            attention_probs = paddle.matmul(kernel_1, self.iterative_inv(kernel_2))
            new_value_layer = paddle.matmul(kernel_3, value_layer)
            context_layer = paddle.matmul(attention_probs, new_value_layer)

        if self.conv_kernel_size is not None:
            context_layer += self.conv(value_layer)

        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class NystromformerSelfOutput(nn.Layer):
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
    def __init__(self, config: NystromformerConfig, position_embedding_type: Optional[str] = None):
        super(NystromformerAttention, self).__init__()
        self.self = NystromformerSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = NystromformerSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None, output_attentions: Optional[bool] = False
    ):
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class NystromformerIntermediate(nn.Layer):
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
    def __init__(self, config: NystromformerConfig):
        super(NystromformerLayer, self).__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = NystromformerAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = NystromformerIntermediate(config)
        self.output = NystromformerOutput(config)

    def apply_chunking_to_forward(
        self, forward_fn: Callable[..., Tensor], chunk_size: int, chunk_dim: int, *input_tensors
    ):
        assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

        # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
        num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
        if num_args_in_forward_chunk_fn != len(input_tensors):
            raise ValueError(
                f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
                "tensors are given"
            )
        if chunk_size > 0:
            tensor_shape = input_tensors[0].shape[chunk_dim]
            for input_tensor in input_tensors:
                if input_tensor.shape[chunk_dim] != tensor_shape:
                    raise ValueError(
                        f"All input tenors have to be of the same shape: {tensor_shape}, "
                        f"found shape {input_tensor.shape[chunk_dim]}"
                    )
            if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
                raise ValueError(
                    f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                    f"size {chunk_size}"
                )
            num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
            input_tensors_chunks = tuple(
                input_tensor.chunk(num_chunks, axis=chunk_dim) for input_tensor in input_tensors
            )
            output_chunks = tuple(
                forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks)
            )
            return paddle.concat(output_chunks, axis=chunk_dim)
        return forward_fn(*input_tensors)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[Tensor] = False,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = self.apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs


class NystromformerEncoder(nn.Layer):
    def __init__(self, config: NystromformerConfig):
        super(NystromformerEncoder, self).__init__()
        self.config = config
        self.layer = nn.LayerList([NystromformerLayer(config) for _ in range(config.num_hidden_layers)])
        # The parameter output_attentions in forward shoule set to be False when self.use_recompute = True.
        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.enable_recompute and self.training:

                def create_cumtom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)[0]

                    return custom_forward

                layer_outputs = (recompute(create_cumtom_forward(layer_module), hidden_states, attention_mask),)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)

            hidden_states = layer_outputs[0]
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
    support_recompute = True

    # model init configuration
    pretrained_init_configuration = NYSTROMFORMER_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = NYSTROMFORMER_PRETRAINED_RESOURCE_FILES_MAP

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding, nn.Conv2D)):
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

    def _set_recompute(self, module, value=False):
        if isinstance(module, NystromformerEncoder):
            module.enable_recompute = value


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
            An instance of ErnieConfig used to construct NystromformerModel.
    """

    def __init__(self, config: NystromformerConfig):
        super(NystromformerModel, self).__init__(config)
        self.embeddings = NystromformerEmbeddings(config)
        self.encoder = NystromformerEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_extended_attention_mask(self, attention_mask, input_shape):
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        The NystromformerModel forward method, overrides the __call__() special method.

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
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions`.

        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import NystromformerModel, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("model_name")
                model = NystromformerModel.from_pretrained("model_name")
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else False

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
                token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

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
        config (:class:`NystromformerConfig`):
            An instance of NystromformerConfig used to construct NystromformerForMaskedLM.
    """

    _keys_to_ignore_on_load_missing = ["cls.predictions.decoder"]

    def __init__(self, config: NystromformerConfig):
        super(NystromformerForMaskedLM, self).__init__(config)

        self.nystromformer = NystromformerModel(config)
        self.cls = NystromformerOnlyMLMHead(config)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], MaskedLMOutput]:
        r"""
        The NystromformerForMaskedLM forward method, overrides the __call__() special method.

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
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., vocab_size]`.
            output_attentions (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.MaskedLMOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.MaskedLMOutput` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.MaskedLMOutput`.

        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import NystromformerForMaskedLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("model_name")
                model = NystromformerForMaskedLM.from_pretrained("model_name")
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
                print(logits.shape) # [batch_size, seq_len, hidden_size]
        """

        return_dict = return_dict if return_dict is not None else False

        outputs = self.nystromformer(
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
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.reshape([-1, self.config.vocab_size]), labels.reshape([-1]))

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
            An instance of ErnieConfig used to construct NystromformerForSequenceClassification.
    """

    def __init__(self, config: NystromformerConfig):
        super(NystromformerForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.nystromformer = NystromformerModel(config)
        self.classifier = NystromformerClassificationHead(config)
        self.config = config if config is not None else NystromformerConfig()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], SequenceClassifierOutput]:
        r"""
        The NystromformerForSequenceClassification forward method, overrides the __call__() special method.

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
            labels (Tensor of shape `(batch_size,)`, optional):
                Labels for computing the sequence classification/regression loss.
                Indices should be in `[0, ..., num_labels - 1]`. If `num_labels == 1`
                a regression loss is computed (Mean-Square loss), If `num_labels > 1`
                a classification loss is computed (Cross-Entropy).
            output_attentions (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.SequenceClassifierOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.SequenceClassifierOutput` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.SequenceClassifierOutput`.

        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import AutoTokenizer, NystromformerForSequenceClassification

                tokenizer = AutoTokenizer.from_pretrained("model_name")
                model = NystromformerForSequenceClassification.from_pretrained("model_name")

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """
        return_dict = return_dict if return_dict is not None else False

        outputs = self.nystromformer(
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
    """
    Nystromformer Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks.

    Args:
        config (:class:`NystromformerConfig`):
            An instance of NystromformerConfig used to construct NystromformerForMultipleChoice.
    """

    def __init__(self, config: NystromformerConfig):
        super(NystromformerForMultipleChoice, self).__init__(config)

        self.nystromformer = NystromformerModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], MultipleChoiceModelOutput]:
        r"""
        The NystromformerForMultipleChoice forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, num_choice, sequence_length].
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type should be int. The `masked` tokens have `0` values and the others have `1` values.
                It is a tensor with shape `[batch_size, num_choice, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Its data type should be `int64` and it has a shape of [batch_size, num_choice, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                ``[0, max_position_embeddings - 1]``.
                Shape as `(batch_size, num_choice, sequence_length)` and dtype as int64. Defaults to `None`.
            inputs_embeds (Tensor, optional):
                Indices of embedded input sequence. They are representations of tokens that build the input sequence.
                Its data type should be `float32` and it has a shape of [batch_size, num_choice, sequence_length, hidden_size].
                Defaults to 'None', which means the input_ids represents the sequence.
            labels (Tensor of shape `(batch_size, )`, optional):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
            output_attentions (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.MultipleChoiceModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.MultipleChoiceModelOutput` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.MultipleChoiceModelOutput`.
        """

        return_dict = return_dict if return_dict is not None else False
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
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]  # (bs * num_choices, seq_len, hidden_size)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, hidden_size)
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, hidden_size)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, hidden_size)
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
    r"""
    Nystromformer Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.
    Args:
        config (:class:`NystromformerConfig`):
            An instance of NystromformerConfig used to construct NystromformerForTokenClassification.
    """

    def __init__(self, config: NystromformerConfig):
        super(NystromformerForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.nystromformer = NystromformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], TokenClassifierOutput]:
        r"""
        The NystromformerForTokenClassification forward method, overrides the __call__() special method.

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
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the token classification loss. Indices should be in `[0, ..., num_labels - 1]`.
            output_attentions (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput`.
        """

        return_dict = return_dict if return_dict is not None else False

        outputs = self.nystromformer(
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
    """
    Nystromformer Model with a linear layer on top of the hidden-states
    output to compute `span_start_logits` and `span_end_logits`,
    designed for question-answering tasks like SQuAD.
    Args:
        config (:class:`NystromformerConfig`):
            An instance of NystromformerConfig used to construct NystromformerForQuestionAnswering.
    """

    def __init__(self, config: NystromformerConfig):
        super(NystromformerForQuestionAnswering, self).__init__(config)

        config.num_labels = 2
        self.nystromformer = NystromformerModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Tensor], QuestionAnsweringModelOutput]:
        r"""
        The NystromformerForMultipleChoice forward method, overrides the __call__() special method.

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
            start_positions (Tensor of shape `(batch_size,)`, optional):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (Tensor of shape `(batch_size,)`, optional):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            output_attentions (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `None`, which means get the option from config.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput`.
        """

        return_dict = return_dict if return_dict is not None else False

        outputs = self.nystromformer(
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

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if end_positions.ndim > 1:
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
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
