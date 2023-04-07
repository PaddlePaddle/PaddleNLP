# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Any, Dict, List

import paddle
import paddle.nn as nn
from paddle.nn import Layer

from paddlenlp.transformers.gpt.modeling import TransformerDecoderLayer
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model

from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from .configuration import (
    OPT_PRETRAINED_INIT_CONFIGURATION,
    OPT_PRETRAINED_RESOURCE_FILES_MAP,
    OPTConfig,
)

__all__ = [
    "OPTModel",
    "OPTPretrainedModel",
    "OPTForCausalLM",
]


class TransformerDecoder(Layer):
    """
    TransformerDecoder is a stack of N decoder layers.
    """

    def __init__(self, config: OPTConfig, decoder_layers: List[Layer]):
        super(TransformerDecoder, self).__init__()

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias_attr=False)
        else:
            self.project_out = None

        self.num_layers = config.num_hidden_layers
        self.layers = decoder_layers

        if config.normalize_before:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.final_layer_norm = None

        self.checkpoints = []

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        use_cache: bool = False,
        cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """
        output = tgt
        new_caches = [] if use_cache else None
        self.checkpoints = []
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, mod in enumerate(self.layers):
            outputs = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                use_cache=use_cache,
                cache=cache[i] if cache is not None else cache,
                output_attentions=output_attentions,
            )

            # outputs = hidden_states if both use_cache and output_attentions are False
            # Otherwise, outputs = (hidden_states, attention if output_attentions, cache if use_cache)
            output = outputs[0] if (use_cache or output_attentions) else outputs

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
            if use_cache:
                new_caches.append(outputs[-1])
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output,)
            self.checkpoints.append(output.name)

        if self.final_layer_norm:
            output = self.final_layer_norm(output)

        if self.project_out:
            output = self.project_out(output)

        if not return_dict:
            temp_list = [output, new_caches, all_hidden_states, all_self_attentions]

            if not (use_cache or output_attentions or output_hidden_states):
                return output

            return tuple(v for v in temp_list if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output,
            past_key_values=new_caches,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )

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
        """OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        and adjust num_embeddings appropriately. Other models don't have this hack.

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

    def __init__(self, config: OPTConfig):
        super(OPTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.word_embed_proj_dim,
            # padding_idx=config.pad_token_id,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=config.initializer_range)),
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias_attr=False)
        else:
            self.project_in = None

        self.position_embeddings = OPTLearnedPositionEmbedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            initializer_range=config.initializer_range,
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, position_ids=None, input_embeddings=None):
        if input_ids is not None:
            input_shape = paddle.shape(input_ids)
            input_embeddings = self.word_embeddings(input_ids)
        else:
            input_shape = paddle.shape(input_embeddings)[:-1]

        if position_ids is None:
            ones = paddle.ones(input_shape, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones

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

    config_class = OPTConfig
    base_model_prefix = "opt"

    pretrained_init_configuration = OPT_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = OPT_PRETRAINED_RESOURCE_FILES_MAP

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
        config (:class:`OPTConfig`):
            An instance of OPTConfig used to construct OPTModel.
    """

    def __init__(self, config: OPTConfig):
        super(OPTModel, self).__init__(config)
        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embeddings = OPTEmbeddings(config)

        config.fuse_attention_qkv = False
        decoder_layers = nn.LayerList()
        for i in range(config.num_hidden_layers):
            decoder_layers.append(TransformerDecoderLayer(config))
        self.decoder = TransformerDecoder(config, decoder_layers)

        self.apply(self.init_weights)
        self.checkpoints = []

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Its data type should be float32.
                The `masked` tokens have `-1e9` values, and the `unmasked` tokens have `0` values.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            inputs_embeds (Tensor, optional):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation
                of shape `(batch_size, sequence_length, hidden_size)`. This is useful if you want more control over
                how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
                Default to None.
            use_cache (bool, optional):
                Whether or not to use cache. Defaults to `False`. If set to `True`, key value states will be returned and
                can be used to speed up decoding.
            cache (list, optional):
                It is a list, and each element in the list is a tuple `(incremental_cache, static_cache)`.
                See `TransformerDecoder.gen_cache <https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/nn/layer/transformer.py#L1060>`__ for more details.
                It is only used for inference and should be None for training.
                Default to `None`.
            output_attentions (bool, optional):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail. Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail. Defaults to `None`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions` object. If `False`, the output
                will be a tuple of tensors. Defaults to `None`.


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

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = paddle.shape(input_ids)
            input_ids = input_ids.reshape((-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = paddle.shape(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        self.checkpoints = []
        past_key_values_length = paddle.shape(cache[0].k)[2] if cache is not None else 0

        if position_ids is None:
            position_ids = paddle.arange(
                past_key_values_length, input_shape[-1] + past_key_values_length, dtype="int64"
            )
            position_ids = position_ids.unsqueeze(0)
            position_ids = paddle.expand(position_ids, input_shape)
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, input_embeddings=inputs_embeds
        )

        # TODO, use registered buffer
        causal_mask = paddle.tensor.triu(paddle.ones((input_shape[-1], input_shape[-1])) * -1e4, diagonal=1)
        if past_key_values_length > 0:
            causal_mask = paddle.concat(
                [
                    paddle.zeros([input_shape[-1], past_key_values_length], dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask + causal_mask
        else:
            attention_mask = causal_mask

        # The tensor returned by triu not in static graph.
        attention_mask.stop_gradient = True

        outputs = self.decoder(
            embedding_output,
            memory=None,
            tgt_mask=attention_mask,
            use_cache=use_cache,
            cache=cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if output_hidden_states:
            if return_dict:
                outputs.hidden_states = (embedding_output,) + outputs.hidden_states
            else:
                # [last_hidden_state, caches, all_hidden_states, all_self_attentions]
                idx = 2 if use_cache else 1
                all_hidden_states = ((embedding_output,) + outputs[idx],)
                outputs = outputs[:idx] + all_hidden_states + outputs[idx + 1 :]

        self.checkpoints.extend(self.decoder.checkpoints)
        return outputs

    def get_input_embeddings(self):
        """get opt input word embedding
        Returns:
            nn.Embedding: the input word embedding of opt mdoel
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, embedding: nn.Embedding):
        """set opt input embedding
        Returns:
            nn.Embedding: the instance of new word embedding
        """
        self.embeddings.word_embeddings = embedding


class OPTLMHead(Layer):
    def __init__(self, hidden_size: int, vocab_size: int, embedding_weights=None):
        super(OPTLMHead, self).__init__()
        self.decoder_weight = (
            self.create_parameter(shape=[vocab_size, hidden_size], dtype=paddle.get_default_dtype(), is_bias=True)
            if embedding_weights is None
            else embedding_weights
        )

    def forward(self, hidden_states):
        if isinstance(hidden_states, BaseModelOutputWithPastAndCrossAttentions):
            hidden_states = hidden_states["last_hidden_state"]

        logits = paddle.tensor.matmul(hidden_states, self.decoder_weight, transpose_y=True)
        return logits


class OPTForCausalLM(OPTPretrainedModel):
    """
    The OPT Model with a `language modeling` head on top.

    Args:
        config (:class:`OPTConfig`):
            An instance of OPTConfig used to construct OPTModel.

    """

    def __init__(self, config: OPTConfig):
        super(OPTForCausalLM, self).__init__(config)
        self.opt = OPTModel(config)
        self.lm_head = OPTLMHead(
            hidden_size=self.opt.config.hidden_size,
            vocab_size=self.opt.config.vocab_size,
            embedding_weights=self.opt.embeddings.word_embeddings.weight,
        )

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`OPTModel`.
            position_ids (Tensor, optional):
                See :class:`OPTModel`.
            attention_mask (Tensor, optional):
                See :class:`OPTModel`.
            inputs_embeds (Tensor, optional):
                See :class:`GPTModel`.
            use_cache (bool, optional):
                See :class:`OPTModel`.
            cache (Tensor, optional):
                See :class:`OPTModel`.
            labels (paddle.Tensor, optional):
                A Tensor of shape `(batch_size, sequence_length)`.
                Labels for language modeling. Note that the labels are shifted inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., vocab_size]`
                Defaults to None.
            output_attentions (bool, optional):
                See :class:`GPTModel`.
            output_hidden_states (bool, optional):
                See :class:`GPTModel`.
            return_dict (bool, optional):
                See :class:`GPTModel`.
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

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.opt(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache=cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs

        logits = self.lm_head(encoder_outputs)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape((-1, shift_logits.shape[-1])), shift_labels.reshape((-1,)))

        if not return_dict:
            if not use_cache:
                return (loss, logits) if loss is not None else logits

            outputs = (logits,) + outputs[1:]
            return ((loss,) + outputs) if loss is not None else outputs

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_fast_entry(self, kwargs: Dict[str, Any]):
        # import FasterOPT at here to avoid cycling import
        from paddlenlp.ops import FasterOPT

        use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        decode_strategy = kwargs.get("decode_strategy")
        # decoding_lib can be passed into FasterOPT
        decoding_lib = kwargs.get("decoding_lib", None)

        if decode_strategy == "beam_search":
            raise AttributeError("'beam_search' is not supported yet in the fast version of OPT")
        # Currently, FasterTransformer only support restricted size_per_head.
        size_per_head = self.opt.config["hidden_size"] // self.opt.config["num_attention_heads"]
        if size_per_head not in [32, 64, 80, 96, 128]:
            raise AttributeError(
                "'size_per_head = %d' is not supported yet in the fast version of OPT" % size_per_head
            )
        if kwargs["forced_bos_token_id"] is not None:
            # not support for forced_bos_token_id yet in the fast version
            raise AttributeError("'forced_bos_token_id != None' is not supported yet in the fast version")
        if kwargs["min_length"] != 0:
            # not support for min_length yet in the fast version
            raise AttributeError("'min_length != 0' is not supported yet in the fast version")
        self._fast_entry = FasterOPT(self, use_fp16_decoding=use_fp16_decoding, decoding_lib=decoding_lib).forward
        return self._fast_entry

    def prepare_inputs_for_generation(self, input_ids, use_cache=False, cache=None, inputs_embeds=None, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask[:, -1, -1, :]
            if "int" in paddle.common_ops_import.convert_dtype(attention_mask.dtype):
                attention_mask = (1.0 - attention_mask) * -1e4
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if cache is not None:
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
                position_ids += 2

        model_inputs.update(
            {
                "cache": cache,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        return model_inputs

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(getattr(self, self.base_model_prefix), name)
