# coding=utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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

import importlib.util
import math
from functools import partial
from typing import ForwardRef

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..model_utils import PretrainedModel, register_base_model

__all__ = [
    "FNetModel",
    "FNetPreTrainedModel",
    "FNetForSequenceClassification"
]


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
            activation_string, list(ACT2FN.keys())))


def mish(x):
    return x * F.tanh(F.softplus(x))


def linear_act(x):
    return x


def swish(x):
    return x * F.sigmoid(x)


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google FNet repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + paddle.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_new": gelu_new,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "mish": mish,
    "linear": linear_act,
    "swish": swish,
}


class FNetPreTrainedModel(PretrainedModel):
    """
    An abstract class for pretrained FNet models. It provides FNet related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    base_model_prefix = "fnet"
    model_config_file = "model_config.json"

    pretrained_init_configuration = {
        "fnet-base": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "pad_token_id": 3,
            "bos_token_id": 1,
            "eos_token_id": 2
        },
        "fnet-large": {
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 4096,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "pad_token_id": 3,
            "bos_token_id": 1,
            "eos_token_id": 2
        }
    }

    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "fnet-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/fnet/fnet-base/model_state.pdparams",
            "fnet-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/fnet/fnet-large/model_state.pdparams",
        }
    }

    base_model_prefix = "fnet"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(mean=0.0,
                                         std=self.initializer_range if hasattr(
                                             self, "initializer_range") else
                                         self.fnet.config["initializer_range"],
                                         shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


class FNetEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self,
                 vocab_size,
                 pad_token_id,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=4):
        super(FNetEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        # NOTE: This is the project layer and will be needed. The original code allows for different embedding and different model dimensions.
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

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
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            position_ids.stop_gradient = True

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.projection(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class FNetIntermediate(nn.Layer):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class FNetBasicOutput(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.layer_norm(input_tensor + hidden_states)
        return hidden_states


class FNetBasicFourierTransform(nn.Layer):
    def __init__(self):
        super().__init__()
        self.fourier_transform = partial(paddle.fft.fftn, axes=(1, 2))

    def forward(self, hidden_states):
        outputs = self.fourier_transform(hidden_states).real()
        return (outputs, )


class FNetFourierTransform(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.self = FNetBasicFourierTransform()
        self.output = FNetBasicOutput(hidden_size)

    def forward(self, hidden_states):
        self_outputs = self.self(hidden_states)
        fourier_output = self.output(self_outputs[0], hidden_states)
        outputs = (fourier_output, )
        return outputs


class FNetOutput(nn.Layer):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class FNetLayer(nn.Layer):
    def __init__(self, hidden_size, intermediate_size, hidden_act,
                 hidden_dropout_prob):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3
        super().__init__()
        self.fourier = FNetFourierTransform(hidden_size)
        self.intermediate = FNetIntermediate(hidden_size, intermediate_size,
                                             hidden_act)
        self.output = FNetOutput(intermediate_size, hidden_size,
                                 hidden_dropout_prob)

    def forward(self, hidden_states):
        self_fourier_outputs = self.fourier(hidden_states)
        fourier_output = self_fourier_outputs[0]
        intermediate_output = self.intermediate(fourier_output)
        layer_output = self.output(intermediate_output, fourier_output)
        return layer_output


class FNetEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.LayerList([
            (encoder_layer if i == 0 else type(encoder_layer)(
                **encoder_layer._config)) for i in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, cache=None):
        output = src
        new_caches = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                output = mod(output)
            else:
                output, new_cache = mod(output, cache=cache[i])
                new_caches.append(new_cache)
        if self.norm is not None:
            output = self.norm(output)
        return output if cache is None else (output, new_caches)


class FNetPooler(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@register_base_model
class FNetModel(FNetPreTrainedModel):
    """
    The model can behave as an encoder, following the architecture described in `FNet: Mixing Tokens with Fourier
    Transforms <https://arxiv.org/abs/2105.03824>`__ by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago
    Ontanon.
    """
    def __init__(self,
                 vocab_size=32000,
                 hidden_size=768,
                 num_hidden_layers=12,
                 intermediate_size=3072,
                 hidden_act="gelu_new",
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=4,
                 initializer_range=0.02,
                 pad_token_id=3,
                 bos_token_id=1,
                 eos_token_id=2,
                 add_pooling_layer=True):
        super(FNetModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = FNetEmbeddings(vocab_size, pad_token_id, hidden_size,
                                         hidden_dropout_prob,
                                         max_position_embeddings,
                                         type_vocab_size)

        encoder_layer = FNetLayer(hidden_size, intermediate_size, hidden_act,
                                  hidden_dropout_prob)
        self.encoder = FNetEncoder(encoder_layer, num_hidden_layers)
        self.pooler = FNetPooler(hidden_size) if add_pooling_layer else None
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                output_hidden_states=False):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )

        embedding_output = self.embeddings(input_ids=input_ids,
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids,
                                           inputs_embeds=inputs_embeds)

        if output_hidden_states:
            output = embedding_output
            encoder_outputs = []
            for mod in self.encoder.layers:
                output = mod(output)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            pooled_output = self.pooler(encoder_outputs[-1])
        else:
            sequence_output = self.encoder(embedding_output)
            pooled_output = self.pooler(sequence_output)
        if output_hidden_states:
            return encoder_outputs, pooled_output
        else:
            return sequence_output, pooled_output


class FNetForSequenceClassification(FNetPreTrainedModel):
    def __init__(self, fnet, num_classes=2, dropout=None):
        super().__init__()
        self.num_labels = num_classes
        self.fnet = fnet

        self.dropout = nn.Dropout(dropout if dropout is not None else self.fnet.
                                  config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.fnet.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        r"""
            The FNetForSequenceClassification forward method, overrides the __call__() special method.

            Args:
                input_ids (Tensor):
                    See :class:`FNetModel`.
                token_type_ids (Tensor, optional):
                    See :class:`FNetModel`.
                position_ids(Tensor, optional):
                    See :class:`FNetModel`.

            Returns:
                Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
                Shape as `[batch_size, num_classes]` and dtype as float32.

            Example:
                .. code-block::

                    import paddle
                    from paddlenlp.transformers.FNet.modeling import FNetForSequenceClassification
                    from paddlenlp.transformers.FNet.tokenizer import FNetTokenizer

                    tokenizer = FNetTokenizer.from_pretrained('fnet-base')
                    model = FNetForSequenceClassification.from_pretrained('fnet-base', num_classes=2)

                    inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                    inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                    logits = model(**inputs)
                    print(logits.shape)
                    # [1, 2]

            """

        _, pooled_output = self.fnet(input_ids,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
