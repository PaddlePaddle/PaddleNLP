from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import paddle


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a regular
    python dictionary.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not paddle.is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (not isinstance(element, (list, tuple)) or
                            not len(element) == 2 or
                            not isinstance(element[0], str)):
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

    def to_tuple(self):
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`paddle.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size,
            1, hidden_size)` is output.
        past_key_values (:obj:`tuple(tuple(paddle.Tensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(paddle.Tensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            ``config.is_encoder_decoder=True`` 2 additional tensors of shape :obj:`(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            ``config.is_encoder_decoder=True`` in the cross-attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`paddle.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` and ``config.add_cross_attention=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class Seq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`paddle.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`tuple(tuple(paddle.Tensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(paddle.Tensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`paddle.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`paddle.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    decoder_attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None
    encoder_last_hidden_state: Optional[paddle.Tensor] = None
    encoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    encoder_attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class Seq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        last_hidden_state (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size,
            1, hidden_size)` is output.
        past_key_values (:obj:`tuple(tuple(paddle.Tensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(paddle.Tensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`paddle.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`paddle.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    last_hidden_state: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    decoder_attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None
    encoder_last_hidden_state: Optional[paddle.Tensor] = None
    encoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
    encoder_attentions: Optional[Tuple[paddle.Tensor]] = None


class Config(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and not k in (
                    "update",
                    "pop", ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [
                self.__class__(x) if isinstance(x, dict) else x for x in value
            ]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Config, self).__setattr__(name, value)
        super(Config, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(Config, self).pop(k, d)
