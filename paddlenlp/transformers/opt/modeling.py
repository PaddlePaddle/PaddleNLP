# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
import random

from typing import Optional

import paddle
from paddle import nn
import numpy as np

from paddle.nn import CrossEntropyLoss
from paddle.nn.layer.transformer import MultiHeadAttention
import paddle.nn.functional as F

from paddlenlp.transformers import PretrainedModel, register_base_model


OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
]


def make_causal_mask(input_ids_shape, dtype, past_key_values_length: int = 0):
    """Make causal mask used for bi-directional self-attention.

    Args:
        input_ids_shape (paddle.Tensor): the shape of input_ids
        dtype : the dtype of mask
        past_key_values_length (int, optional): the length of past key values. Defaults to 0.

    Returns:
        Tensor: causal mask of shape [bath_size, 1, seq_len, seq_len + past_key_values_length].
    """
    bsz, tgt_len = input_ids_shape
    mask = paddle.full(shape=(tgt_len, tgt_len), fill_value=float("-inf"))
    zero_mask = paddle.full(shape=(tgt_len, tgt_len), fill_value=0)

    mask_cond = paddle.arange(mask.shape[-1])
    condition = mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1])
    mask = paddle.where(condition, zero_mask, mask)
    mask = paddle.cast(mask, dtype=dtype)

    if past_key_values_length > 0:
        mask = paddle.concat([paddle.zeros(shape=[tgt_len, past_key_values_length], dtype=dtype), mask], axis=-1)
    return mask[None, None, :, :].expand(shape=[bsz, 1, tgt_len, tgt_len + past_key_values_length])


def get_finfo(dtype) -> np.finfo:
    """get the info by paddle dtype

    Args:
        dtype : paddle dtype

    Raises:
        ValueError: if dtype is not one of float16, float32, float64

    Returns:
        np.finfo: the np.finfo
    """
    float_dtype_maps = {
        paddle.float64: np.float64,
        paddle.float32: np.float32,
        paddle.float16: np.float16
    }
    if dtype not in float_dtype_maps:
        raise ValueError(f'dtype<{dtype}> should be one of {float_dtype_maps.keys()}')

    return np.finfo(float_dtype_maps[dtype])


def _expand_mask(mask, dtype, tgt_len: Optional[int] = None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.

    Args:
        mask (paddle.Tensor): the source of attention mask
        dtype : _description_
        tgt_len (Optional[int], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(shape=[bsz, 1, tgt_len, src_len])
    expanded_mask = paddle.cast(expanded_mask, dtype)

    inverted_mask = 1.0 - expanded_mask

    min_value_mask = paddle.full_like(inverted_mask, fill_value=get_finfo(dtype).min)

    # TODO: Question: pytorch中的mask_fill在paddle就只能通过where来实现的话，此时就需要创建两个tensor，感觉有些空间浪费，有没有更高效一点的方法
    return paddle.where(inverted_mask > 0, min_value_mask, inverted_mask)


def make_positions(mask, padding_idx: int):
    """Replace non-padding symbols with their position numbers.
    
    Position numbers begin at padding_idx+1. Padding symbols are ignored.

    Args:
        mask (paddle.Tensor): the tensor of maek
        padding_idx (int): id of padding

    Returns:
        paddle.Tensor: new mask with positions
    """
    cum_mask = paddle.cumsum(mask, axis=1)
    return paddle.cast(cum_mask, mask.dtype) * mask + padding_idx


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size. Padding ids are ignored by either offsetting
    based on padding_idx or by setting padding_idx to None and ensuring that the appropriate position ids are passed to
    the forward function.

    Args:
        num_embeddings (int): the number of embedding
        embedding_dim (int): the hidden size of embedding
        padding_idx (int, optional): the id of padding index. Defaults to 1.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 1):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self._padding_idx is not None:
            self.max_positions = self._num_embeddings - self._padding_idx - 1
        else:
            self.max_positions = self._num_embeddings

    def forward(self, attention_mask, positions = None):
        if not ((positions is None) or (self._padding_idx is None)):
            raise ValueError("If positions is pre-computed then padding_idx should not be set.")

        if positions is None:
            attention_mask = paddle.cast(attention_mask, paddle.int64)
            positions = make_positions(attention_mask, self._padding_idx)

        return F.embedding(
            x=positions,
            weight=self.weight,
            padding_idx=self._padding_idx,
            sparse=self._sparse,
            name=self._name,
        )


class OPTDecoderLayer(nn.Layer):
    """

    Args:
        embed_dim (int, optional): _description_. Defaults to 768.
        num_attention_heads (int, optional): _description_. Defaults to 12.
        attention_dropout (float, optional): _description_. Defaults to 0.0.
        do_layer_norm_before (bool, optional): _description_. Defaults to True.
        dropout (float, optional): _description_. Defaults to 0.0.
        activation_function (str, optional): _description_. Defaults to 'relu'.
        activation_dropout (float, optional): _description_. Defaults to 0.0.
        ffn_dim (int, optional): _description_. Defaults to 3072.
    """    
    def __init__(
        self,
        embed_dim: int = 768,
        num_attention_heads: int = 12,
        attention_dropout: float = 0.0,
        do_layer_norm_before: bool = True,
        dropout: float = 0.0,
        activation_function: str = 'relu',
        activation_dropout: float = 0.0,
        ffn_dim: int = 3072,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            need_weights=True
        )
        self.do_layer_norm_before = do_layer_norm_before
        self.dropout = dropout
        self.activation_fn = getattr(F, activation_function)

        self.activation_dropout = activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        # TODO: layer head mask
        layer_head_mask = None,
        output_attentions = False,
        use_cache = False,
        past_key_value = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # TODO: past key-value to be tested
        hidden_states, self_attn_weights = self.self_attn(
            query=hidden_states,
            attn_mask=attention_mask,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(shape=[-1, hidden_states.shape[-1]])
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).reshape(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class OPTPreTrainedModel(PretrainedModel):
    # TODO: 合适起作用
    base_model_prefix = "opt"
    # TODO: to check if support gradient checkpoint
    supports_gradient_checkpointing = True
    # ignore: 
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]


    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "facebook/opt-125m": {
            "activation_dropout": 0.0,
            "activation_function": "relu",
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "hidden_size": 768,
            "do_layer_norm_before": True,
            "dropout": 0.1,
            "eos_token_id": 2,
            "ffn_dim": 3072,
            "init_std": 0.02,
            "layerdrop": 0.0,
            "max_position_embeddings": 2048,
            "model_type": "opt",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "torch_dtype": "float16",
            "use_cache": True,
            "vocab_size": 50272,
            "word_embed_proj_dim": 768,
            "prefix": "</s>"
        },
        "facebook/opt-350m": {},
        "facebook/opt-1.3b": {},
        "facebook/opt-2.7b": {},
        "facebook/opt-6.7b": {},
        "facebook/opt-13b": {},
        "facebook/opt-30b": {},
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "facebook/opt-125m":
            # TODO: to upload the pdparams to the cloud
            "https://bj.bcebos.com/paddlenlp/models/transformers/bert-base-uncased.pdparams",
        }
    }

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.model.config["init_std"],
                        shape=layer.weight.shape))


class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(
        self,
        vocab_size=50272,
        hidden_size=768,
        num_hidden_layers=12,
        ffn_dim=3072,
        max_position_embeddings=2048,
        do_layer_norm_before=True,
        word_embed_proj_dim=None,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        num_attention_heads=12,
        activation_function="relu",
        layerdrop=0.0,
        pad_token_id=1,
        
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        use_return_dict: bool = False
    ):
        super().__init__()
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.padding_idx = pad_token_id
        self.max_target_positions = max_position_embeddings
        self.vocab_size = vocab_size
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_cache = use_cache
        self.use_return_dict = use_return_dict

        self.embed_tokens = nn.Embedding(vocab_size, word_embed_proj_dim, self.padding_idx)

        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        if self.padding_idx is not None:
            num_embeddings = max_position_embeddings + 2

        self.embed_positions = OPTLearnedPositionalEmbedding(num_embeddings, hidden_size, self.padding_idx)

        if word_embed_proj_dim != hidden_size:
            self.project_out = nn.Linear(hidden_size, word_embed_proj_dim, bias_attr=False)
        else:
            self.project_out = None

        if word_embed_proj_dim != hidden_size:
            self.project_in = nn.Linear(word_embed_proj_dim, hidden_size, bias_attr=False)
        else:
            self.project_in = None

        self.layer_norm = None
        self.layers = nn.LayerList([OPTDecoderLayer(
            embed_dim= hidden_size,
            num_attention_heads = num_attention_heads,
            attention_dropout = attention_dropout,
            do_layer_norm_before = do_layer_norm_before,
            dropout = dropout,
            activation_function = activation_function,
            activation_dropout = activation_dropout,
            ffn_dim = ffn_dim
        ) for _ in range(num_hidden_layers)])

        self.gradient_checkpointing = False
        
        # TODO: paddle 中是否有post init功能
        # Initialize weights and apply final processing

        # self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`OPTTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions or self.output_attentions
        output_hidden_states = output_hidden_states or self.output_hidden_states
        # TODO: enable use_cache later
        # use_cache = use_cache or self.use_cache
        use_cache = False

        return_dict = return_dict or self.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.reshape([-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            # to check the
            attention_mask = paddle.ones(inputs_embeds.shape[:2], dtype=paddle.bool)

        positions = self.embed_positions(attention_mask)[:, past_key_values_length:, :]

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.shape[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.shape[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    # TODO: use warning tools
                    # logger.warning(
                    #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    # )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        # TODO: what's the output structure of paddlenlp
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }


@register_base_model
class OPTModel(OPTPreTrainedModel):
    def __init__(
        self,
        vocab_size=50272,
        hidden_size=768,
        num_hidden_layers=12,
        ffn_dim=3072,
        max_position_embeddings=2048,
        do_layer_norm_before=True,
        word_embed_proj_dim=None,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        num_attention_heads=12,
        activation_function="relu",
        layerdrop=0.0,
        init_std=0.02,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        use_return_dict: bool = False
    ):
        super().__init__()
        self.decoder = OPTDecoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            max_position_embeddings=max_position_embeddings,
            do_layer_norm_before=do_layer_norm_before,
            word_embed_proj_dim=word_embed_proj_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            num_attention_heads=num_attention_heads,
            activation_function=activation_function,
            layerdrop=layerdrop,
            pad_token_id=pad_token_id,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            use_cache = use_cache,
            use_return_dict = use_return_dict   
        )

        # Initialize weights and apply final processing
        # TODO: post init
        # self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):

        output_attentions = output_attentions or self.decoder.config["output_attentions"]
        output_hidden_states = output_hidden_states or self.decoder.config["output_hidden_states"]
        
        use_cache = use_cache or self.decoder.config["use_cache"]
        return_dict = return_dict or self.decoder.config["use_return_dict"]

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return decoder_outputs
        # return BaseModelOutputWithPast(
        #     last_hidden_state=decoder_outputs.last_hidden_state,
        #     past_key_values=decoder_outputs.past_key_values,
        #     hidden_states=decoder_outputs.hidden_states,
        #     attentions=decoder_outputs.attentions,
        # )


class OPTForCausalLM(OPTPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head\.weight"]

    def __init__(self, model: OPTModel):
        super().__init__()
        self.model = model

        self.lm_head_weight = model.decoder.embed_tokens.weight

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`OPTTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import OPTTokenizer, OPTForCausalLM
        # this needs fixing

        >>> tokenizer = OPTTokenizer.from_pretrained("patrickvonplaten/opt_gpt2_tokenizer")
        >>> model = OPTForCausalLM.from_pretrained("ArthurZ/opt-350m")
        >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
        >>> list(logits.shape) == expected_shape
        True
        ```"""
        if len(attention_mask.shape) == 4:
            attention_mask_shape = attention_mask.shape
            assert attention_mask_shape[1] == 1
            assert attention_mask_shape[2] == 1
            attention_mask = attention_mask.squeeze(1).squeeze(1)

        output_attentions = output_attentions or self.model.config["output_attentions"]
        output_hidden_states = output_hidden_states or self.model.config["output_hidden_states"]
        
        return_dict = return_dict or self.model.config["use_return_dict"]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


    # def forward(self, hidden_states, masked_positions=None):
    #     if masked_positions is not None:
    #         hidden_states = paddle.reshape(
    #             hidden_states, [-1, paddle.shape(hidden_states)[-1]])
    #         hidden_states = paddle.gather(hidden_states, masked_positions)
    #     # gather masked tokens might be more quick
    #     hidden_states = self.transform(hidden_states)
    #     hidden_states = self.activation(hidden_states)
    #     hidden_states = self.layer_norm(hidden_states)

        logits = paddle.matmul(outputs[0], self.lm_head_weight, transpose_y=True)
        
        # logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(logits.reshape([-1, self.model.config["vocab_size"]]), labels.reshape(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return {
            "loss": loss,
            "logits": logits,
            # TODO: 从其它模块借鉴
            "hidden_states": outputs.hidden_states,
        }

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = paddle.ones(input_ids.shape)

        if past:
            input_ids = input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
