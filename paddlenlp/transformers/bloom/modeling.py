# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
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
"""PyTorch BLOOM model."""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import paddle
from paddle import Tensor, nn
from paddle.autograd import PyLayer
from paddle.fluid.framework import in_dygraph_mode

from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from paddlenlp.transformers.model_utils import PretrainedModel

from ...utils.converter import StateDictNameMapping
from ...utils.log import logger
from .configuration import BloomConfig

BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigscience/bigscience-small-testing",
    "bigscience/bloom-350m",
    "bigscience/bloom-760m",
    "bigscience/bloom-1b3",
    "bigscience/bloom-2b5",
    "bigscience/bloom-6b3",
    "bigscience/bloom",
]


def split_tensor_along_last_dim(tensor: Tensor, num_partitions: int, contiguous_split_chunks: bool = False):
    """Split a tensor along its last dimension -> query/key/value layer
    Args:
        tensor: ([`paddle.Tensor`], *required*):
            input tensor to split
        num_partitions ([`int`], *required*):
            number of partitions to split the tensor
        contiguous_split_chunks ([`bool`], *optional*, default=`False`)::
            If True, make each chunk contiguous in memory.
    """
    return paddle.split(tensor, 3, axis=-1)


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def attention_mask_func(attention_scores: Tensor, attention_mask: Tensor, causal_mask: Tensor):

    # attention_mask_bool = ~attention_mask.bool()
    attention_mask_bool = ~paddle.cast(attention_mask, dtype=paddle.bool)

    query_length, key_length, n_heads = attention_scores.shape[2], attention_scores.shape[3], attention_scores.shape[1]
    padded_causal_mask = paddle.logical_or(
        attention_mask_bool[:, None, key_length - query_length : key_length, None],
        ~paddle.cast(causal_mask[:, :, key_length - query_length : key_length, :key_length], dtype=paddle.bool),
    )
    padded_causal_mask = paddle.logical_or(padded_causal_mask, attention_mask_bool[:, None, None, :key_length])
    # Make use of floats
    padded_causal_mask.stop_gradient = True
    attention_scores = masked_fill(attention_scores, padded_causal_mask.expand([-1, n_heads, -1, -1]), -10000.0)
    return (
        attention_scores,
        padded_causal_mask,
    )


def build_alibi_tensor(max_seq_len: int, n_head: int, dtype=paddle.bfloat16):
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742

    Args:
    Returns tensor shaped (n_head, 1, max_seq_len)
     hdiwipa   max_seq_len: (`int`, *required*):
            max sequence length
        n_head: (`int`, *required*):
            number of heads
        dtype: (`paddle.dtype`, *optional*, default=`paddle.bfloat16`):
            dtype of the output tensor
    """

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = paddle.to_tensor(get_slopes(n_head), dtype=dtype).unsqueeze(1).unsqueeze(1)
    arange_tensor = paddle.arange(max_seq_len).unsqueeze(0).unsqueeze(0)
    alibi = slopes * arange_tensor.expand([n_head, -1, -1])

    alibi = paddle.cast(alibi, dtype=dtype)

    return alibi


def pre_process_alibi_for_pad(alibi, attention_mask, num_heads):
    """
    Args:
    Pre-process the alibi tensor for padding.
        alibi: ([`torch.tensor`], *required*):
            alibi tensor to pre-process
        attention_mask: ([`torch.tensor`], *required*):
            attention mask to pre-process"""

    # Sanity check if we are not inferring less tokens than the total sequence length
    # This usually happens when the inference is done with past_key_values
    # In this case we re-create the alibi tensor with the correct sequence length
    if attention_mask.shape[-1] != alibi.shape[-1]:
        alibi = build_alibi_tensor(attention_mask.shape[-1], num_heads, alibi.dtype).repeat_interleave(
            attention_mask.shape[0], axis=0
        )
    # Get the indexes of the padding tokens
    index_x0, index_y0 = paddle.where(attention_mask == 0.0)
    index_x1, index_y1 = paddle.where(attention_mask == 1.0)

    # Clone the embeddings  - we can detach because the embeddings are not learned
    # Get a refence tensor
    slice_reference_alibi = build_alibi_tensor(alibi.shape[-1], num_heads, alibi.dtype)

    # Loop over the batch where the padding is and replace the alibi tensor by the reference tensor
    # Only where you do not have padding. Replace padding tokens by zeros
    # This operation can be seen as a shifting operation.
    for i, index in enumerate(paddle.unique(index_x0)):
        slice_to_modify = paddle.zeros_like(slice_reference_alibi)
        index_shift = index_y1[index_x1 == index]
        shift_value = len(index_shift)
        slice_to_modify[:, :, index_shift] = slice_reference_alibi[:, :, :shift_value]
        alibi[index * num_heads : (index + 1) * num_heads] = slice_to_modify
    return alibi


def dropout_add(x, residual, prob, training):
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *rquired*):
            esidual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def bloom_gelu_forward(x):
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + paddle.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def bloom_gelu_back(g, x):
    """
    gradient of tanh approximation of gelu gradient of actual gelu is: 0.5 * (1. + torch.erf(x * 0.70710678)) +
    0.3989423 * x * torch.exp(-0.5 * x * x)

    Args:
        g (`torch.tensor`, *required*):
            gradient output tensor
        x (`torch.tensor`, *required*):
            input tensor
    """
    x = x[0]  # x is a tuple of 1 element, needs to unpack it first
    tanh_out = paddle.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


def baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0):
    return beta * input + alpha * paddle.bmm(batch1, batch2)


# class GeLUFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return bloom_gelu_forward(input)

#     @staticmethod
#     def backward(ctx, grad_output):
#         input = ctx.saved_tensors
#         tmp = bloom_gelu_back(grad_output, input)
#         return tmp


class GeLUFunction(PyLayer):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = bloom_gelu_back(grad_output, input)
        return tmp


class BloomGelu(nn.Layer):
    """
    BloomBiasGelu wrapper function that make use of the simple function on inference mode to make the model
    torchscriptable and use the autograd function in training mode to get the accurate results of the gradients Partly
    copied from Megatron-DeepSpeed code and adapted for our needs

    See here why autograd functions are not torchscriptable: https://github.com/pytorch/pytorch/issues/22329

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):

        if self.training and in_dygraph_mode():
            return GeLUFunction.apply(x)
        else:
            return bloom_gelu_forward(x)


class BloomScaledSoftmax(nn.Layer):
    """
    fused operation: scaling + mask + softmax

    Args:
        input_in_fp16 (`bool`, *required*):
            flag to indicate if input in fp16 data format.
        input_in_bf16 (`bool`, *required*):
            flag to indicate if input in bf16 data format.
        scaled_masked_softmax_fusion (`bool`, *required*):
            flag to indicate user want to use softmax fusion
        mask_func (`function`, *required*):
            mask function to be applied.
        softmax_in_fp32 (`bool`, *required*):
            if true, softmax in performed at fp32 precision.
        scale (`float`, *required*):
            scaling factor used in input tensor scaling.
    """

    def __init__(self, scaled_masked_softmax_fusion, mask_func, softmax_in_fp32, scale):
        super().__init__()
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        if not (self.scale is None or softmax_in_fp32):
            raise ValueError("softmax should be in fp32 when scaled")

    def forward(self, input, mask, max_positions):
        input_dtype = input.dtype
        input_in_16bit = input_dtype in [paddle.float16, paddle.bfloat16]
        softmax_dtype = paddle.float32 if self.softmax_in_fp32 else input_dtype

        if self.scale is not None:
            input = input * self.scale

        if mask is None:
            mask = paddle.ones([input.shape[0], max_positions])

        seq_ids = paddle.arange(max_positions)
        causal_mask = (seq_ids[None, :] <= seq_ids[:, None]).reshape([1, 1, max_positions, max_positions])
        mask_output, padded_causal_mask = self.mask_func(input, mask, causal_mask)
        probs = nn.functional.softmax(mask_output, axis=-1, dtype=softmax_dtype) * (~padded_causal_mask)

        if input_in_16bit and self.softmax_in_fp32:
            probs = paddle.cast(probs, dtype=input_dtype)

        return probs


class BloomAttention(nn.Layer):
    def __init__(self, config, layer_number=None):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.masked_softmax_fusion = config.masked_softmax_fusion
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.layer_number = max(1, layer_number)
        self.norm_factor = math.sqrt(self.head_dim) * self.layer_number

        # Scaled Softmax
        self.scale_mask_softmax = BloomScaledSoftmax(
            self.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            self.layer_number,
        )

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias_attr=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states,
        residual,
        layer_past=None,
        attention_mask=None,
        alibi=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]
        # repeat alibi tensor with the batch size

        alibi = alibi.repeat_interleave(hidden_states.shape[0], axis=0)
        # alibi = alibi.repeat(hidden_states.shape[0], 1, 1).to(hidden_states.device)

        # apply preprocessing if the input is padded
        if attention_mask is not None and 0 in attention_mask:
            alibi = pre_process_alibi_for_pad(alibi, attention_mask, self.num_heads)

        mixed_x_layer = self.query_key_value(hidden_states)

        # [batch_size, seq_length, 3 x hidden_size] --> [batch_size, seq_length, num_heads, 3 x head_dim]
        new_tensor_shape = mixed_x_layer.shape[:-1] + [self.num_heads, 3 * self.head_dim]
        mixed_x_layer = mixed_x_layer.reshape(new_tensor_shape)

        # [batch_size, seq_length, num_heads, 3 x head_dim] --> 3  [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = paddle.cat((past_key.type_as(key_layer), key_layer), axis=1)
            value_layer = paddle.cat((past_value.type_as(value_layer), value_layer), axis=1)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [batch_size, head_dim, q_length, k_length]
        output_size = (query_layer.shape[0], query_layer.shape[2], query_layer.shape[1], key_layer.shape[1])

        # [batch_size, q_length, num_heads, head_dim] -> [q_length, batch_size * num_heads, head_dim]
        query_layer = query_layer.transpose([1, 0, 2, 3]).reshape(
            [output_size[2], output_size[0] * output_size[1], -1]
        )

        # [batch_size, k_length, num_heads, head_dim] -> [k_length, batch_size * num_heads, head_dim]
        key_layer = key_layer.transpose([1, 0, 2, 3]).reshape([output_size[3], output_size[0] * output_size[1], -1])

        # slice alibi tensor until the query length
        sliced_alibi = alibi[: output_size[0] * output_size[1], :, : output_size[3]]

        # Raw attention scores. [batch_size * num_heads, q_length, k_length]
        beta = 1.0 / self.layer_number

        matmul_result = baddbmm(
            sliced_alibi,
            query_layer.transpose([1, 0, 2]),
            key_layer.transpose([1, 0, 2]).transpose([0, 2, 1]),
            beta=beta,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [batch_size, num_heads, q_length, k_length]
        attention_scores = matmul_result.reshape([*output_size])

        # attention scores and attention mask [b, np, sq, sk]
        max_positions = max(attention_scores.shape[-1], attention_scores.shape[-2])
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask, max_positions)
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context layer shape: [batch_size, num_heads, q_length, head_dim]
        output_size = (value_layer.shape[0], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])

        # change view [k_length, batch_size x num_heads, head_dim]
        value_layer = value_layer.transpose([1, 0, 2, 3]).reshape(
            [value_layer.shape[1], output_size[0] * output_size[1], -1]
        )

        # change view [batch_size x num_heads, q_length, k_length]
        attention_probs_reshaped = attention_probs.reshape([output_size[0] * output_size[1], output_size[2], -1])

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = paddle.bmm(attention_probs_reshaped, value_layer.transpose([1, 0, 2]))

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = context_layer.reshape([*output_size])

        # [batchs_size, num_heads, q_length, head_dim] --> [q_length, batch_size, num_heads, head_dim]
        # context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        context_layer = context_layer.transpose([2, 0, 1, 3])

        # [q_length, batch_size, num_heads, head_dim] --> [q_length, batch_size, hidden_size]
        new_context_layer_shape = context_layer.shape[:-2] + [self.hidden_size]

        context_layer = context_layer.reshape([*new_context_layer_shape])

        # Output. [q_length, batch_size, hidden_size]

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = context_layer.shape[-1] / self.pretraining_tp
            output_tensor = paddle.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + nn.functional.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output = output_tensor.transpose([1, 0, 2])

        output = dropout_add(output, residual, self.hidden_dropout, self.training)

        outputs = (output, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class BloomMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
        self.hidden_dropout = config.hidden_dropout
        self.gelu_impl = BloomGelu()

    def forward(self, hidden_states, residual):
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = paddle.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + nn.functional.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)

        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

        return output


class BloomBlock(nn.Layer):
    def __init__(self, config, layer_number=None):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = nn.LayerNorm(hidden_size, epsilon=config.layer_norm_epsilon)
        self.n_head = config.n_head
        self.self_attention = BloomAttention(config, layer_number=layer_number)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, epsilon=config.layer_norm_epsilon)

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        alibi=None,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class BloomPreTrainedModel(PretrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BloomConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BloomBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """Initialize the weights."""
        # TODO(wj-Mcat): init for embedding
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.set_value(
                paddle.tensor.normal(mean=0.0, std=self.initializer_range, shape=module.weight.shape)
            )
            if getattr(module, "bias", None) is not None:
                module.weight.set_value(paddle.tensor.zeros(shape=module.weight.shape))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BloomModel):
            module.gradient_checkpointing = value

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"

        head_mask = paddle.cast(head_mask, dtype=self.dtype)
        return head_mask

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    @classmethod
    def _get_name_mappings(cls, config: BloomConfig) -> list[StateDictNameMapping]:
        hard_mapping = [
            ["word_embeddings.weight", "word_embeddings.weight"],
            ["word_embeddings_layernorm.weight", "word_embeddings_layernorm.weight"],
            ["word_embeddings_layernorm.bias", "word_embeddings_layernorm.bias"],
            ["ln_f.weight", "ln_f.weight"],
            ["ln_f.bias", "ln_f.bias"],
        ]
        for i in range(config.n_layer):
            hard_mapping.extend(
                [
                    [f"h.{i}.input_layernorm.weight", f"h.{i}.input_layernorm.weight"],
                    [f"h.{i}.input_layernorm.bias", f"h.{i}.input_layernorm.bias"],
                    [
                        f"h.{i}.self_attention.query_key_value.weight",
                        f"h.{i}.self_attention.query_key_value.weight",
                        "transpose",
                    ],
                    [f"h.{i}.self_attention.query_key_value.bias", f"h.{i}.self_attention.query_key_value.bias"],
                    [f"h.{i}.self_attention.dense.weight", f"h.{i}.self_attention.dense.weight", "transpose"],
                    [f"h.{i}.self_attention.dense.bias", f"h.{i}.self_attention.dense.bias"],
                    [f"h.{i}.post_attention_layernorm.weight", f"h.{i}.post_attention_layernorm.weight"],
                    [f"h.{i}.post_attention_layernorm.bias", f"h.{i}.post_attention_layernorm.bias"],
                    [f"h.{i}.mlp.dense_h_to_4h.weight", f"h.{i}.mlp.dense_h_to_4h.weight", "transpose"],
                    [f"h.{i}.mlp.dense_h_to_4h.bias", f"h.{i}.mlp.dense_h_to_4h.bias"],
                    [f"h.{i}.mlp.dense_4h_to_h.weight", f"h.{i}.mlp.dense_4h_to_h.weight", "transpose"],
                    [f"h.{i}.mlp.dense_4h_to_h.bias", f"h.{i}.mlp.dense_4h_to_h.bias"],
                ]
            )
        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(hard_mapping)]
        return mappings


class BloomModel(
    BloomPreTrainedModel,
):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.n_head = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        self.word_embeddings_layernorm = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_epsilon)

        # Transformer blocks
        self.h = nn.LayerList([BloomBlock(config, layer_number=i) for i in range(config.n_layer)])

        # Final Layer Norm
        self.ln_f = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[Tensor], BaseModelOutputWithPastAndCrossAttentions]:
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
            input_ids = input_ids.reshape([-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_head x N x N
        # head_mask has shape n_layer x batch x n_head x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        output_shape = input_shape + [hidden_states.shape[-1]]

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        current_sequence_length = hidden_states.shape[1]
        if past_key_values[0] is not None:
            current_sequence_length += past_key_values[0][0].shape[1]
        alibi = build_alibi_tensor(current_sequence_length, self.n_head, hidden_states.dtype)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions, alibi)

                    return custom_forward

                outputs = paddle.distributed.fleet.utils.recompute(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = hidden_states.reshape(output_shape)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BloomForCausalLM(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape([-1, shift_logits.shape[-1]]), shift_labels.reshape([-1]))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[Tensor]], beam_idx: Tensor) -> Tuple[Tuple[Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class BloomForSequenceClassification(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = BloomModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias_attr=False)

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
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
                sequence_lengths = paddle.where(input_ids != self.config.pad_token_id, 1, 0).sum(axis=-1) - 1
                # sequence_lengths = paddle.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[paddle.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype == paddle.int:
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
                loss = loss_fct(pooled_logits.reshape([-1, self.num_labels]), labels.reshape([-1]))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
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


class BloomForTokenClassification(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = BloomModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.reshape([-1]))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
