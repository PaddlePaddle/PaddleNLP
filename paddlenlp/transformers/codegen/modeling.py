# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The Salesforce authors, The Open AI Team Authors and The HuggingFace Inc. team
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

from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer

from ...utils.env import CONFIG_NAME
from ...utils.log import logger
from .. import PretrainedModel, register_base_model
from ..activations import ACT2FN
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from .configuration import (
    CODEGEN_PRETRAINED_INIT_CONFIGURATION,
    CODEGEN_PRETRAINED_RESOURCE_FILES_MAP,
    CodeGenConfig,
)

CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/codegen-350M-nl",
    "Salesforce/codegen-350M-multi",
    "Salesforce/codegen-350M-mono",
    "Salesforce/codegen-2B-nl",
    "Salesforce/codegen-2B-multi",
    "Salesforce/codegen-2B-mono",
    "Salesforce/codegen-6B-nl",
    "Salesforce/codegen-6B-multi",
    "Salesforce/codegen-6B-mono",
    "Salesforce/codegen-16B-nl",
    "Salesforce/codegen-16B-multi",
    "Salesforce/codegen-16B-mono",
]


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2) / dim))
    sinusoid_inp = paddle.einsum("i,j->ij", paddle.arange(seq_len, dtype="float32"), inv_freq)
    return paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = paddle.stack((-x2, x1), axis=-1)
    # In einsum notation: rearrange(x, '... d j -> ... (d j)')
    return x.flatten(-2)


def duplicate_interleave(m):
    return paddle.repeat_interleave(m, 2, axis=1)


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(lambda t: duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :], sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class CodeGenAttention(Layer):
    def __init__(self, config: CodeGenConfig):
        super().__init__()

        self.causal_mask = paddle.tril(
            paddle.ones((config.n_positions, config.n_positions), dtype=paddle.get_default_dtype())
        ).reshape((1, 1, config.n_positions, config.n_positions))

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.n_embd
        self.num_attention_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = paddle.sqrt(paddle.to_tensor(self.head_dim, dtype="float32"))
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias_attr=False)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=False)
        self.rotary_dim = config.rotary_dim

    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + [n_head // mp_num, dim_head])
        reshaped = reshaped.reshape(x.shape[:-2] + [-1] + reshaped.shape[-1:])
        return reshaped

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        if len(tensor.shape) == 5:
            tensor = tensor.transpose([0, 1, 3, 2, 4])
        elif len(tensor.shape) == 4:
            tensor = tensor.transpose([0, 2, 1, 3])
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.shape[:-2] + [num_attention_heads * attn_head_size]
        return tensor.reshape(new_shape)

    def _attn(self, query, key, value, attention_mask=None):

        # compute causal mask from causal mask buffer
        query_length, key_length = query.shape[-2], key.shape[-2]
        causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = paddle.cast(query, "float32")
        key = paddle.cast(key, "float32")
        attn_weights = paddle.matmul(query, key, transpose_y=True)

        attn_weights = attn_weights / self.scale_attn
        mask_value = paddle.to_tensor(-1e4, dtype=attn_weights.dtype)
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        attn_weights = paddle.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, axis=-1, dtype=value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = paddle.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: Optional[bool] = False,
        cache: Optional[Tuple[Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple:
        qkv = self.qkv_proj(hidden_states)
        mp_num = 4
        qkv_split = qkv.reshape(qkv.shape[:-1] + [mp_num, -1])

        local_dim = qkv_split.shape[-1] // (self.head_dim * self.num_attention_heads // mp_num)
        query, value, key = paddle.split(qkv_split, local_dim, axis=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.transpose([0, 2, 1, 3])

        seq_len = key.shape[1]
        offset = 0

        if cache is not None:
            offset = cache[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = paddle.concat([k_rot, k_pass], axis=-1)
            query = paddle.concat([q_rot, q_pass], axis=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.transpose([0, 2, 1, 3])
        query = query.transpose([0, 2, 1, 3])

        if cache is not None:
            past_key = cache[0]
            past_value = cache[1]
            key = paddle.concat((past_key, key), axis=-2)
            value = paddle.concat((past_value, value), axis=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        if output_attentions:
            return attn_output, present, attn_weights
        return attn_output, present


class CodeGenMLP(Layer):
    def __init__(self, config: CodeGenConfig):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.fc_in = nn.Linear(config.n_embd, inner_dim)
        self.fc_out = nn.Linear(inner_dim, config.n_embd)

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CodeGenBlock(Layer):
    def __init__(self, config: CodeGenConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, epsilon=config.layer_norm_epsilon)
        self.attn = CodeGenAttention(config)
        self.mlp = CodeGenMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: Optional[bool] = False,
        cache: Optional[Tuple[Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            cache=cache,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, (present, attentions)  outputs is a tuple


class CodeGenPreTrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    model_config_file = CONFIG_NAME
    pretrained_init_configuration = CODEGEN_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = CODEGEN_PRETRAINED_RESOURCE_FILES_MAP
    config_class = CodeGenConfig
    base_model_prefix = "transformer"

    def _init_weights(self, layer):
        """Initialize the weights."""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            if isinstance(layer.weight, paddle.Tensor) and paddle.get_default_dtype() == "float32":
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.full_like(layer.weight, 1.0))
            layer._epsilon = self.config.layer_norm_epsilon
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            layer.bias.set_value(paddle.zeros_like(layer.bias))


@register_base_model
class CodeGenModel(CodeGenPreTrainedModel):
    r"""
    The bare CodeGen Model outputting raw hidden-states.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.
    Args:
        config (:class:`CodeGenConfig`):
            An instance of CodeGenConfig used to construct CodeGenModel.
    """

    def __init__(self, config: CodeGenConfig):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.bos_token_id = config.bos_token_id
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.embed_dim = config.n_embd
        self.initializer_range = config.initializer_range
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.LayerList([CodeGenBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_epsilon)
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.n_head)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        cache: Optional[List[Tuple[Tensor]]] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        The CodeGenModel forward method, overrides the `__call__()` special method.
        Args:
            input_ids (Tensor, optional):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Defaults to `None`, which means nothing needed to be prevented attention to.
            use_cache (bool, optional):
                 Whether or not to use cache. Defaults to `False`. If set to `True`, key value states will be returned and
                 can be used to speed up decoding.
            cache (list, optional):
                It is a list, and each element in the list is a tuple `(incremental_cache, static_cache)`.
                See `TransformerDecoder.gen_cache <https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/nn/layer/transformer.py#L1060>`__ for more details.
                It is only used for inference and should be None for training.
                Default to `None`.
            inputs_embeds (Tensor, optional):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation
                of shape `(batch_size, sequence_length, hidden_size)`. This is useful if you want more control over
                how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
                Default to None.
            output_attentions (bool, optional):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail. Defaults to `False`.
            output_hidden_states (bool, optional):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail. Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions` object.
                If `False`, the output will be a tuple of tensors. Defaults to `False`.
        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions`.
            Especially, When `return_dict=output_hidden_states=output_attentions=False` and `cache=None`,
            returns a tensor representing the output of :class:`CodeGenModel`.
            Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import CodeGenModel, CodeGenTokenizer
                tokenizer = CodeGenTokenizer.from_pretrained('Salesforce/codegen-350M-mono')
                model = CodeGenModel.from_pretrained('Salesforce/codegen-350M-mono')
                inputs = tokenizer("def hello_world():", return_token_type_ids=False)
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """
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
            input_ids = input_ids.reshape((-1, input_shape[-1]))
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if cache is None:
            past_length = 0
            cache = tuple([None] * len(self.h))
        else:
            past_length = cache[0][0].shape[-2]

        # Attention mask.
        if attention_mask is None:
            if input_ids is not None:
                if batch_size == 1 and past_length != 0:
                    batch_size, seq_len = input_shape
                    attention_mask = paddle.zeros(
                        [batch_size, 1, 1, seq_len + past_length], dtype=paddle.get_default_dtype()
                    )
                else:
                    attention_mask = (
                        paddle.cast(input_ids == self.pad_token_id, dtype=paddle.get_default_dtype()).unsqueeze([1, 2])
                        * -1e4
                    )
            else:
                logger.warning(
                    "Provided inputs_embeds while attention_mask is None, attention weights will not be masked during forwarding."
                )
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
        if attention_mask is not None:
            attention_mask.stop_gradient = True
        # TODO: CodeGen Attention Mask is TOO confusion.
        # When it's 2D, it must be int and it's denoted by 1/0.
        # When using model.generate() without providing attention mask
        # or using 4D attention mask,
        # the attention mask's dtype must be float and it's denoted by 0/-inf.
        # Moreover, cannot support 3D attention mask.

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            inputs_embeds = inputs_embeds + token_type_embeds

        hidden_states = self.drop(inputs_embeds)
        output_shape = input_shape[:] + [hidden_states.shape[-1]]

        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, (block, old_cache) in enumerate(zip(self.h, cache)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                cache=old_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions += (outputs[-1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.reshape(shape=output_shape)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        last_hidden_state = hidden_states
        new_cache = presents

        if not return_dict:
            temp_list = [
                last_hidden_state,
                new_cache,
                all_hidden_states,
                all_self_attentions,
            ]
            return tuple(v for v in temp_list if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            past_key_values=new_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )


class CodeGenForCausalLM(CodeGenPreTrainedModel):
    r"""
    CodeGen Model with a `language modeling` head on top.
    Args:
        config (:class:`CodeGenConfig`):
            An instance of CodeGenConfig used to construct CodeGenForCausalLM.
    """
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"h\.\d+\.attn\.bias"]

    def __init__(self, config: CodeGenConfig):
        super().__init__(config)
        self.transformer = CodeGenModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_fast_entry(self, kwargs):
        from paddlenlp.ops import FasterCodeGen

        use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        decoding_lib = kwargs.get("decoding_lib", None)
        decode_strategy = kwargs.get("decode_strategy")
        if decode_strategy == "beam_search":
            raise AttributeError("'beam_search' is not supported yet in the fast version of GPTJ")
        # Currently, FasterTransformer only support restricted size_per_head.
        size_per_head = self.transformer.config.n_embd // self.transformer.config.n_head
        if size_per_head not in [32, 64, 80, 96, 128, 160, 192, 224, 256]:
            raise AttributeError(
                "'size_per_head = %d' is not supported yet in the fast version of GPTJ" % size_per_head
            )
        if kwargs["forced_bos_token_id"] is not None:
            # not support for min_length yet in the fast version
            raise AttributeError("'forced_bos_token_id != None' is not supported yet in the fast version")
        self._fast_entry = FasterCodeGen(self, decoding_lib=decoding_lib, use_fp16_decoding=use_fp16_decoding).forward
        return self._fast_entry

    def prepare_inputs_for_generation(self, input_ids, cache=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        token_type_ids = kwargs.get("token_type_ids", None)

        if cache:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask[:, :, -1:, :]

        return {
            "input_ids": input_ids,
            "cache": cache,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        cache: Optional[List[Tuple[Tensor]]] = None,
        labels: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        The CodeGenForCausalLM forward method, overrides the __call__() special method.
        Args:
            input_ids (Tensor, optional):
                See :class:`CodeGenModel`.
            attention_mask (Tensor, optional):
                See :class:`CodeGenModel`.
            use_cache (bool, optional):
                See :class:`CodeGenModel`.
            cache (Tensor, optional):
                See :class:`CodeGenModel`.
            labels: (Tensor, optional):
                Labels for language modeling. Note that the labels are shifted inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., vocab_size]`
            inputs_embeds (Tensor, optional):
                See :class:`CodeGenModel`.
            output_attentions (bool, optional):
                See :class: `CodeGenModel`.
            output_hidden_states (bool, optional):
                See :class: `CodeGenModel`.
            return_dict (bool, optional):
                See :class: `CodeGenModel`.
        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.CausalLMOutputWithPastAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.CausalLMOutputWithPastAndCrossAttentions`.
            Especially, When `return_dict=output_hidden_states=output_attentions=False` and `cache=labels=None`,
            returns tensor `lm_logits` of shape [batch_size, sequence_length, vocab_size],

        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import CodeGenForCausalLM, CodeGenTokenizer
                tokenizer = CodeGenTokenizer.from_pretrained('Salesforce/codegen-350M-mono')
                model = CodeGenForCausalLM.from_pretrained('Salesforce/codegen-350M-mono')
                inputs = tokenizer("def hello_world():", return_token_type_ids=False)
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_cache=use_cache,
            cache=cache,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = paddle.cast(self.lm_head(hidden_states), "float32")

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape((-1, shift_logits.shape[-1])), shift_labels.reshape((-1,)))

        if not return_dict:
            # if isinstance(transformer_outputs, type(input_ids)):
            #     return (loss, lm_logits) if loss is not None else lm_logits
            outputs = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + outputs) if loss is not None else outputs

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(getattr(self, self.base_model_prefix), name)
