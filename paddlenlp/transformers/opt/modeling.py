# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
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

import collections
from typing import Optional, List, Dict, Any

import paddle
import paddle.nn as nn
from paddle.nn import LayerNorm, Layer
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.fluid import layers
from paddle.nn.layer.transformer import _convert_param_attr_to_list
from paddlenlp.transformers.gpt.modeling import MultiHeadAttention, TransformerDecoderLayer
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model

__all__ = [
    "OPTModel",
    "OPTPretrainedModel",
    "OPTForCausalLM",
]


class TransformerDecoder(Layer):
    """
    TransformerDecoder is a stack of N decoder layers.
    """

    def __init__(
        self,
        decoder_layers: List[Layer],
        num_layers: int,
        hidden_size: int,
        word_embed_proj_dim: int,
        norm: Optional[Layer] = None,
        normalize_before: bool = False,
    ):
        super(TransformerDecoder, self).__init__()

        if word_embed_proj_dim != hidden_size:
            self.project_out = nn.Linear(hidden_size, word_embed_proj_dim, bias_attr=False)
        else:
            self.project_out = None

        self.num_layers = num_layers
        self.layers = decoder_layers

        if normalize_before:
            self.final_layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.final_layer_norm = None

        self.checkpoints = []

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, use_cache: bool = False, cache=None):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """
        output = tgt
        new_caches = []
        self.checkpoints = []

        for i, mod in enumerate(self.layers):
            if cache is None:
                if use_cache:
                    output, new_cache = mod(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache)
                    new_caches.append(new_cache)
                else:
                    output = mod(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache)

            else:
                output, new_cache = mod(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache[i])
                new_caches.append(new_cache)
            self.checkpoints.append(output.name)

        if self.final_layer_norm:
            output = self.final_layer_norm(output)

        if self.project_out:
            output = self.project_out(output)

        return output if use_cache is False else (output, new_caches)

    def gen_cache(self, memory, do_zip=False):
        r"""
        Generates cache for `forward` usage. The generated cache is a list, and
        each element in it is a tuple( :code:`(incremental_cache, static_cache)` )
        produced by `TransformerDecoderLayer.gen_cache`. See `TransformerDecoderLayer.gen_cache`
        for more details. If `do_zip` is True, apply `zip` on these tuples to get
        a list with two elements.
        """
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache


class OPTLearnedPositionEmbedding(nn.Embedding):
    """this module learns postional embeddings up to a fixed maximum size"""

    def __init__(self, num_embeddings: int, embedding_dim: int, initializer_range: float):
        """OPT is set up so taht if padding_idx is specified then offset the embedding ids by 2
        and adjust num_embeddings appropriately. Other models don't have this hack

        Args:
            num_embeddings (int): the number of embedding size
            embedding_dim (int): the dim of embedding
        """
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, position_ids, past_key_values_length: int = 0):
        """get the position embedding with attention mask

        Args:
            position_ids: (paddle.Tensor): the tensor of position ids
            past_key_values_length (int, optional): the past key value which will . Defaults to 0.

        Returns:
            paddle.Tensor: the position embedding
        """
        # cut positions if `past_key_values_length` is > 0
        position_ids = position_ids[:, past_key_values_length:]
        return super().forward(position_ids + self.offset)


class OPTEmbeddings(Layer):
    """
    Include embeddings from word and position embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        word_embed_proj_dim: int = 768,
        padding_idx: int = 1,
        hidden_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: Optional[int] = None,
        initializer_range=0.02,
    ):
        super(OPTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size,
            word_embed_proj_dim,
            # padding_idx=padding_idx,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)),
        )

        if word_embed_proj_dim != hidden_size:
            self.project_in = nn.Linear(word_embed_proj_dim, hidden_size, bias_attr=False)
        else:
            self.project_in = None

        self.position_embeddings = OPTLearnedPositionEmbedding(
            num_embeddings=max_position_embeddings, embedding_dim=hidden_size, initializer_range=initializer_range
        )

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones

        input_embeddings = self.word_embeddings(input_ids)

        if self.project_in:
            input_embeddings = self.project_in(input_embeddings)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class OPTPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained OPT models. It provides OPT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    pretrained_init_configuration = {}
    pretrained_resource_files_map = {"model_state": {}}
    base_model_prefix = "opt"

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
                        if hasattr(self, "initializer_range")
                        else self.opt.config["initializer_range"],
                        shape=layer.weight.shape,
                    )
                )


@register_base_model
class OPTModel(OPTPretrainedModel):
    r"""
    The bare OPT Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `OPTModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `OPTModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer and decoder layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer decoder. Defaults to `12`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer decoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the decoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `"relu"`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and decoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all decoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids`. Defaults to `16`.

            .. note::
                Please NOT using `type_vocab_size`, for it will be obsolete in the future..

        initializer_range (float, optional):
            The standard deviation of the normal initializer. Default to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`OPTPretrainedModel._init_weights()` for how weights are initialized in `OPTModel`.

        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
             to `0`.

    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        word_embed_proj_dim: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "relu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 16,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        eos_token_id: int = 7,
        bos_token_id: int = 0,
        eol_token_id: int = 3,
        normalize_before: bool = True,
        **kwargs
    ):
        super(OPTModel, self).__init__()

        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embeddings = OPTEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            word_embed_proj_dim=word_embed_proj_dim,
            padding_idx=pad_token_id,
            hidden_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
        )

        decoder_layers = nn.LayerList()
        for i in range(num_hidden_layers):
            decoder_layers.append(
                TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=intermediate_size,
                    dropout=hidden_dropout_prob,
                    activation=hidden_act,
                    attn_dropout=attention_probs_dropout_prob,
                    act_dropout=hidden_dropout_prob,
                    weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(mean=0.0, std=self.initializer_range)
                    ),
                    bias_attr=None,
                    normalize_before=normalize_before,
                )
            )

        self.decoder = TransformerDecoder(
            decoder_layers,
            num_hidden_layers,
            norm="LayerNorm",
            hidden_size=hidden_size,
            normalize_before=normalize_before,
            word_embed_proj_dim=word_embed_proj_dim,
        )

        self.apply(self.init_weights)
        self.checkpoints = []

    def forward(self, input_ids, position_ids=None, attention_mask=None, use_cache=False, cache=None):
        r"""
        The OPTModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in self attention to avoid performing attention to some unwanted positions,
                usually the subsequent positions.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Its data type should be float32.
                The `masked` tokens have `-1e-9` values, and the `unmasked` tokens have `0` values.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            use_cache (bool, optional):
                Whether or not to use cache. Defaults to `False`. If set to `True`, key value states will be returned and
                can be used to speed up decoding.
            cache (list, optional):
                It is a list, and each element in the list is a tuple `(incremental_cache, static_cache)`.
                See `TransformerDecoder.gen_cache <https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/nn/layer/transformer.py#L1060>`__ for more details.
                It is only used for inference and should be None for training.
                Default to `None`.

        Returns:
            Tensor: Returns tensor `encoder_output`, which is the output at the last layer of the model.
            Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import OPTModel, GPTTokenizer

                tokenizer = GPTTokenizer.from_pretrained('facebook/opt-125m')

                model = OPTModel.from_pretrained('facebook/opt-125m')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLimage.pngP!", return_token_type_ids=False)
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """

        self.checkpoints = []
        if position_ids is None:
            past_length = 0
            if cache is not None:
                past_length = paddle.shape(cache[0].k)[-2]
            position_ids = paddle.arange(past_length, paddle.shape(input_ids)[-1] + past_length, dtype=input_ids.dtype)
            position_ids = position_ids.unsqueeze(0)

            position_ids = paddle.expand_as(position_ids, input_ids)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # TODO, use registered buffer
        causal_mask = paddle.tensor.triu(
            paddle.ones((paddle.shape(input_ids)[-1], paddle.shape(input_ids)[-1])) * -1e4, diagonal=1
        )

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask + causal_mask
        else:
            attention_mask = causal_mask
        # The tensor returned by triu not in static graph.
        attention_mask.stop_gradient = True

        decoder_outputs = self.decoder(
            embedding_output, memory=None, tgt_mask=attention_mask, use_cache=use_cache, cache=cache
        )

        self.checkpoints.extend(self.decoder.checkpoints)
        return decoder_outputs


class OPTLMHead(Layer):
    def __init__(self, hidden_size: int, vocab_size: int, embedding_weights=None):
        super(OPTLMHead, self).__init__()
        self.decoder_weight = (
            self.create_parameter(shape=[vocab_size, hidden_size], dtype=paddle.get_default_dtype(), is_bias=True)
            if embedding_weights is None
            else embedding_weights
        )

    def forward(self, hidden_states):
        logits = paddle.tensor.matmul(hidden_states, self.decoder_weight, transpose_y=True)
        return logits


class OPTForCausalLM(OPTPretrainedModel):
    """
    The OPT Model with a `language modeling` head on top.

    Args:
        opt (:class:`OPTModel`):
            An instance of :class:`OPTModel`.

    """

    def __init__(self, opt: OPTModel):
        super(OPTForCausalLM, self).__init__()
        self.opt = opt
        self.lm_head = OPTLMHead(
            hidden_size=self.opt.config["hidden_size"],
            vocab_size=self.opt.config["vocab_size"],
            embedding_weights=self.opt.embeddings.word_embeddings.weight,
        )

    def forward(self, input_ids, position_ids=None, attention_mask=None, use_cache=False, cache=None):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`OPTModel`.
            position_ids (Tensor, optional):
                See :class:`OPTModel`.
            attention_mask (Tensor, optional):
                See :class:`OPTModel`.
            use_cache (bool, optional):
                See :class:`OPTModel`.
            cache (Tensor, optional):
                See :class:`OPTModel`.

        Returns:
            Tensor or tuple: Returns tensor `logits` or tuple `(logits, cached_kvs)`. If `use_cache` is True,
            tuple (`logits, cached_kvs`) will be returned. Otherwise, tensor `logits` will be returned.
            `logits` is the output of the opt model.
            `cache_kvs` is the cache output of opt model if `use_cache` is True.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import OPTForCausalLM, GPTTokenizer

                tokenizer = GPTTokenizer.from_pretrained('facebook/opt-125m')
                model = OPTForCausalLM.from_pretrained('facebook/opt-125m')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output_ids, score = model.generate(input_ids=inputs['input_ids'])
                print(tokenizer.batch_decode(output_ids[0]))
        """

        outputs = self.opt(
            input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=use_cache, cache=cache
        )

        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs

        logits = self.lm_head(encoder_outputs)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits

    def prepare_faster_entry(self, kwargs: Dict[str, Any]):
        # import FasterOPT at here to avoid cycling import
        from paddlenlp.ops import FasterOPT

        use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        decode_strategy = kwargs.get("decode_strategy")
        # decoding_lib can be passed into FasterOPT
        decoding_lib = kwargs.get("decoding_lib", None)

        if decode_strategy == "beam_search":
            raise AttributeError("'beam_search' is not supported yet in the faster version of OPT")
        # Currently, FasterTransformer only support restricted size_per_head.
        size_per_head = self.opt.config["hidden_size"] // self.opt.config["num_attention_heads"]
        if size_per_head not in [32, 64, 80, 96, 128]:
            raise AttributeError(
                "'size_per_head = %d' is not supported yet in the faster version of OPT" % size_per_head
            )
        if kwargs["forced_bos_token_id"] is not None:
            # not support for min_length yet in the faster version
            raise AttributeError("'forced_bos_token_id != None' is not supported yet in the faster version")
        if kwargs["min_length"] != 0:
            # not support for min_length yet in the faster version
            raise AttributeError("'min_length != 0' is not supported yet in the faster version")
        self._faster_entry = FasterOPT(self, use_fp16_decoding=use_fp16_decoding, decoding_lib=decoding_lib).forward
        return self._faster_entry

    def prepare_inputs_for_generation(self, input_ids, use_cache=False, cache=None, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask[:, -1, -1, :]
            if "int" in paddle.common_ops_import.convert_dtype(attention_mask.dtype):
                attention_mask = (1.0 - attention_mask) * -1e4
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
                position_ids += 2
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache,
        }

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            try:
                return getattr(getattr(self, self.base_model_prefix), name)
            except AttributeError:
                try:
                    return getattr(self, self.base_model_prefix).config[name]
                except KeyError:
                    raise e
