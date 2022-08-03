# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import functools
import paddle
import numpy as np
from collections import OrderedDict
from dataclasses import fields, dataclass
from typing import Any, List, Tuple, Optional
from paddle.nn.layer.transformer import _convert_attention_mask

from .utils import adapt_stale_fwd_patch


def _transformer_encoder_layer_fwd(self,
                                   src,
                                   src_mask=None,
                                   cache=None,
                                   output_attentions=False):
    self.self_attn.need_weights = output_attentions
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    residual = src
    if self.normalize_before:
        src = self.norm1(src)

    attn_outputs = self.self_attn(src, src, src, src_mask, cache)
    if isinstance(attn_outputs, tuple):
        src = attn_outputs[0]
        outputs = attn_outputs[1:]
    else:
        src = attn_outputs
        outputs = None

    src = residual + self.dropout1(src)
    if not self.normalize_before:
        src = self.norm1(src)

    residual = src
    if self.normalize_before:
        src = self.norm2(src)
    src = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = residual + self.dropout2(src)
    if not self.normalize_before:
        src = self.norm2(src)

    return src if outputs is None else (
        (src, ) + outputs[::-1])  # hidden_states, cache, attentions


def _transformer_encoder_fwd(self,
                             src,
                             src_mask=None,
                             cache=None,
                             output_attentions=False,
                             output_hidden_states=False,
                             return_dict=False):
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    output = src
    new_caches = [] if cache is not None else None
    all_attentions = [] if output_attentions else None
    # NOTE: Also includes embeding output which is same as HF.
    all_hidden_states = [output] if output_hidden_states else None
    for i, mod in enumerate(self.layers):
        layer_outputs = mod(output,
                            src_mask=src_mask,
                            cache=None if cache is None else cache[i],
                            output_attentions=output_attentions)
        if isinstance(layer_outputs, tuple):
            output = layer_outputs[0]
            outputs = layer_outputs[1:]
        else:
            output = layer_outputs
            outputs = None

        if output_hidden_states:
            all_hidden_states.append(output)
        if output_attentions:
            all_attentions.append(outputs[-1])
        if cache is not None:
            new_caches.append(outputs[0])

    if self.norm is not None:
        output = self.norm(output)

        if output_hidden_states:
            all_hidden_states[-1] = output

    if not return_dict:
        outputs = tuple(
            tuple(v) if isinstance(v, list) else v for v in [
                output,
                new_caches,
                all_hidden_states,
                all_attentions,
            ] if v is not None)
        if len(outputs) == 1:
            return output
        else:
            return outputs

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=output,
        past_key_values=new_caches,
        hidden_states=all_hidden_states,
        attentions=all_attentions)


# patches of paddle.nn.Transformer to get all hidden_states and attentions
paddle.nn.TransformerEncoderLayer.forward = _transformer_encoder_layer_fwd
paddle.nn.TransformerEncoder.forward = _transformer_encoder_fwd


def _get_wrap_setattr(cls):

    def _wrap_setattr(self, name, value):
        value = adapt_stale_fwd_patch(self, name, value)
        return super(cls, self).__setattr__(name, value)

    return _wrap_setattr


paddle.nn.TransformerEncoderLayer.__setattr__ = functools.wraps(
    paddle.nn.TransformerEncoderLayer.__setattr__)(_get_wrap_setattr(
        paddle.nn.TransformerEncoderLayer))
paddle.nn.TransformerEncoder.__setattr__ = functools.wraps(
    paddle.nn.TransformerEncoder.__setattr__)(_get_wrap_setattr(
        paddle.nn.TransformerEncoder))


def is_tensor(x):
    if isinstance(x, paddle.Tensor):
        return True

    return isinstance(x, np.ndarray)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # note(guosheng): Convert list to tuple automatically, and better to
        # check if it is frozen.
        # assert not getattr(self, dataclasses._PARAMS).frozen
        for f in class_fields:
            value = getattr(self, f.name)
            if isinstance(value, list):
                setattr(self, f.name, tuple(value))

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(
                f"{self.__class__.__name__} should not have more than one required field."
            )

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (not isinstance(element,
                                       (list, tuple)) or not len(element) == 2
                            or not isinstance(element[0], str)):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`paddle.Tensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(paddle.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """

    last_hidden_state: paddle.Tensor = None
    pooler_output: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class SequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`paddle.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class TokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    start_logits: paddle.Tensor = None
    end_logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class MultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.

    Args:
        loss (`paddle.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`paddle.Tensor` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

            Classification scores (before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
