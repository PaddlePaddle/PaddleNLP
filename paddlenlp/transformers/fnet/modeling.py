# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modeling classes for FNet model."""

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from functools import partial
from paddle.nn import Layer
from .. import PretrainedModel, register_base_model

__all__ = [
    "FNetBasicOutput",
    "FNetOutput",
    "FNetIntermediate",
    "FNetLayer",
    "FNetEncoder",
    "FNetPooler",
    "FNetEmbeddings",
    "FNetBasicFourierTransform",
    "FNetFourierTransform",
    "FNetPredictionHeadTransform",
    "FNetLMPredictionHead",
    "FNetOnlyMLMHead",
    "FNetOnlyNSPHead",
    "FNetPreTrainingHeads",
    "FNetPretrainedModel",
    "FNetModel",
    "FNetForSequenceClassification",
    "FNetForPreTraining",
    "FNetForMaskedLM",
    "FNetForNextSentencePrediction",
    "FNetForMultipleChoice",
    "FNetForTokenClassification",
    "FNetForQuestionAnswering"
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
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + paddle.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))


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


class FNetBasicOutput(Layer):
    def __init__(self, hidden_size, layer_norm_eps):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.layer_norm(input_tensor + hidden_states)
        return hidden_states


class FNetOutput(Layer):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 layer_norm_eps,
                 hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(input_tensor + hidden_states)
        return hidden_states


class FNetIntermediate(Layer):
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


class FNetLayer(Layer):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 layer_norm_eps,
                 hidden_dropout_prob,
                 hidden_act):
        super().__init__()
        self.fourier = FNetFourierTransform(hidden_size, layer_norm_eps)
        self.intermediate = FNetIntermediate(hidden_size, intermediate_size, hidden_act)
        self.output = FNetOutput(hidden_size,
                                 intermediate_size,
                                 layer_norm_eps,
                                 hidden_dropout_prob)
    
    def forward(self, hidden_states):
        self_fourier_outputs = self.fourier(hidden_states)
        fourier_output = self_fourier_outputs[0]
        intermediate_output = self.intermediate(fourier_output)
        layer_output = self.output(intermediate_output, fourier_output)
        
        return (layer_output,)


class FNetEncoder(Layer):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 layer_norm_eps,
                 hidden_dropout_prob,
                 hidden_act,
                 num_hidden_layers):
        super().__init__()
        self.layers = nn.LayerList([FNetLayer(hidden_size,
                                              intermediate_size,
                                              layer_norm_eps,
                                              hidden_dropout_prob,
                                              hidden_act) for _ in range(num_hidden_layers)])
        self.gradient_checkpointing = False
    
    def forward(self, hidden_states, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if return_dict:
            return {"last_hidden_state": hidden_states,
                    "all_hidden_states": all_hidden_states
                    }
        return (hidden_states,)


class FNetPooler(Layer):
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


class FNetEmbeddings(Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    
    def __init__(
            self,
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
            pad_token_id,
    ):
        super(FNetEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        # NOTE: This is the project layer and will be needed. The original code allows for different embedding and different model dimensions.
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", paddle.arange(max_position_embeddings).expand((1, -1)))
    
    def forward(
            self,
            input_ids,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.projection(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class FNetBasicFourierTransform(Layer):
    def __init__(self):
        super().__init__()
        # self.fourier_transform = paddle.fft.fftn
        self.fourier_transform = partial(paddle.fft.fftn, axes=(1, 2))
    
    def forward(self, hidden_states):
        outputs = self.fourier_transform(hidden_states).real()
        return (outputs,)


class FNetFourierTransform(Layer):
    def __init__(self, hidden_size, layer_norm_eps):
        super().__init__()
        self.fourier_transform = FNetBasicFourierTransform()
        self.output = FNetBasicOutput(hidden_size, layer_norm_eps)
    
    def forward(self, hidden_states):
        self_outputs = self.fourier_transform(hidden_states)
        fourier_output = self.output(self_outputs[0], hidden_states)
        return (fourier_output,)


class FNetPredictionHeadTransform(Layer):
    def __init__(self, hidden_size, layer_norm_eps, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = ACT2FN[hidden_act]
        else:
            self.transform_act_fn = hidden_act
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class FNetLMPredictionHead(Layer):
    def __init__(self, hidden_size, vocab_size, layer_norm_eps, hidden_act):
        super().__init__()
        self.transform = FNetPredictionHeadTransform(hidden_size, layer_norm_eps, hidden_act)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
        self.bias = self.create_parameter(
            [vocab_size],
            is_bias=True,
            default_initializer=nn.initializer.Constant(value=0)
        )
        self.decoder.bias = self.bias
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class FNetOnlyMLMHead(Layer):
    def __init__(self, hidden_size, vocab_size, layer_norm_eps, hidden_act):
        super().__init__()
        self.predictions = FNetLMPredictionHead(hidden_size, vocab_size, layer_norm_eps, hidden_act)
    
    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class FNetOnlyNSPHead(Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.seq_relationship = nn.Linear(hidden_size, 2)
    
    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class FNetPreTrainingHeads(Layer):
    def __init__(self, hidden_size, vocab_size, layer_norm_eps, hidden_act):
        super().__init__()
        self.predictions = FNetLMPredictionHead(hidden_size, vocab_size, layer_norm_eps, hidden_act)
        self.seq_relationship = nn.Linear(hidden_size, 2)
    
    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class FNetPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained FNet models. It provides FNet related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """
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
            "layer_norm_eps": 1e-12,
            "pad_token_id": 3,
            "bos_token_id": 1,
            "eos_token_id": 2,
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
            "layer_norm_eps": 1e-12,
            "pad_token_id": 3,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "fnet-base": "https://huggingface.co/HJHGJGHHG/paddle-fnet-base/resolve/main/model_state.pdparams",
            "fnet-large": "https://huggingface.co/HJHGJGHHG/paddle-fnet-large/resolve/main/model_state.pdparams"
        }
    }
    base_model_prefix = "fnet"
    
    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, layer):
        # Initialize the weights.
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.fnet.config["initializer_range"],
                    shape=layer.weight.shape)
            )
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.fnet.config["initializer_range"],
                    shape=layer.weight.shape)
            )
            if layer._padding_idx is not None:
                layer.weight[layer._padding_idx].set_value(
                    paddle.zeros_like(layer.weight[layer._padding_idx])
                )
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.ones_like(layer.weight))


@register_base_model
class FNetModel(FNetPretrainedModel):
    """
    The model can behave as an encoder, following the architecture described in `FNet: Mixing Tokens with Fourier
    Transforms <https://arxiv.org/abs/2105.03824>`__ by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago
    Ontanon.
    """
    
    def __init__(
            self,
            vocab_size=32000,
            hidden_size=768,
            num_hidden_layers=12,
            intermediate_size=3072,
            hidden_act="gelu_new",
            hidden_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=4,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=3,
            bos_token_id=1,
            eos_token_id=2,
            add_pooling_layer=True
    ):
        super(FNetModel, self).__init__()
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.embeddings = FNetEmbeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
            pad_token_id
        )
        self.encoder = FNetEncoder(
            hidden_size,
            intermediate_size,
            layer_norm_eps,
            hidden_dropout_prob,
            hidden_act,
            num_hidden_layers
        )
        self.pooler = FNetPooler(hidden_size) if add_pooling_layer else None
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if token_type_ids is None:
            token_type_ids = paddle.zeros(shape=input_shape, dtype="int64")
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        pooler_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        if return_dict:
            return {"last_hidden_state": sequence_output,
                    "pooler_output": pooler_output,
                    "all_hidden_states": encoder_outputs["all_hidden_states"]
                    }
        return (sequence_output, pooler_output) + encoder_outputs[1:]


class FNetForSequenceClassification(FNetPretrainedModel):
    def __init__(self, fnet, num_classes=2):
        super(FNetForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.fnet = fnet
        
        self.dropout = nn.Dropout(self.fnet.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.fnet.config["hidden_size"], num_classes)
        
        # Initialize weights and apply final processing
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": outputs["all_hidden_states"],
            }
        return logits


class FNetForPreTraining(FNetPretrainedModel):
    def __init__(self, fnet):
        super().__init__()
        
        self.fnet = fnet
        self.cls = FNetPreTrainingHeads(
            self.fnet.config["hidden_size"],
            self.fnet.config["vocab_size"],
            self.fnet.config["layer_norm_eps"],
            self.fnet.config["hidden_act"])
        
        self.init_weights()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def get_input_embeddings(self):
        return self.fnet.embeddings.word_embeddings
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0] if not return_dict \
            else outputs["last_hidden_state"]
        pooled_output = outputs[1] if not return_dict \
            else outputs["pooler_output"]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        
        if return_dict:
            return {
                "prediction_logits": prediction_scores,
                "seq_relationship_logits": seq_relationship_score,
                "hidden_states": outputs["all_hidden_states"]
            }
        return prediction_scores, seq_relationship_score, outputs["all_hidden_states"]


class FNetForMaskedLM(FNetPretrainedModel):
    def __init__(self, fnet):
        super().__init__()
        
        self.fnet = fnet
        self.cls = FNetOnlyMLMHead(
            self.fnet.config["hidden_size"],
            self.fnet.config["vocab_size"],
            self.fnet.config["layer_norm_eps"],
            self.fnet.config["hidden_act"])
        
        self.init_weights()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def get_input_embeddings(self):
        return self.fnet.embeddings.word_embeddings
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] if not return_dict \
            else outputs["last_hidden_state"]
        prediction_scores = self.cls(sequence_output)
        
        if return_dict:
            return {
                "prediction_logits": prediction_scores,
                "hidden_states": outputs["all_hidden_states"]
            }
        return prediction_scores, outputs["all_hidden_states"]


class FNetForNextSentencePrediction(FNetPretrainedModel):
    def __init__(self, fnet):
        super().__init__()
        
        self.fnet = fnet
        self.cls = FNetOnlyNSPHead(self.fnet.config["hidden_size"])
        
        self.init_weights()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def get_input_embeddings(self):
        return self.fnet.embeddings.word_embeddings
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1] if not return_dict \
            else outputs["pooler_output"]
        seq_relationship_score = self.cls(pooled_output)
        
        if return_dict:
            return {
                "seq_relationship_logits": seq_relationship_score,
                "hidden_states": outputs["all_hidden_states"]
            }
        return seq_relationship_score, outputs["all_hidden_states"]


class FNetForMultipleChoice(FNetPretrainedModel):
    def __init__(self, fnet):
        super(FNetForMultipleChoice, self).__init__()
        self.fnet = fnet
        self.dropout = nn.Dropout(self.fnet.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.fnet.config["hidden_size"], 1)
        
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        input_ids = input_ids.reshape([-1, input_ids.shape[-1]]) \
            if input_ids is not None else None
        token_type_ids = token_type_ids.reshape([-1, token_type_ids.shape[-1]]) \
            if token_type_ids is not None else None
        position_ids = position_ids.reshape([-1, position_ids.shape[-1]]) \
            if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.reshape([-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1]])
            if inputs_embeds is not None
            else None
        )
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1] if not return_dict else outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape([-1, num_choices])
        if return_dict:
            return {
                "logits": reshaped_logits,
                "hidden_states": outputs["all_hidden_states"],
            }
        return reshaped_logits


class FNetForTokenClassification(FNetPretrainedModel):
    def __init__(self, fnet, num_classes=2):
        super(FNetForTokenClassification, self).__init__()
        self.fnet = fnet
        self.num_classes = num_classes
        self.dropout = nn.Dropout(self.fnet.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.fnet.config["hidden_size"], self.num_classes)
        
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] if not return_dict else outputs["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": outputs["all_hidden_states"],
            }
        return logits


class FNetForQuestionAnswering(FNetPretrainedModel):
    def __init__(self, fnet, num_labels):
        super(FNetForQuestionAnswering, self).__init__()
        self.num_labels = num_labels
        self.fnet = fnet
        self.qa_outputs = nn.Linear(self.fnet.config["hidden_size"], num_labels)
        
        self.init_weights()
    
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] if not return_dict else outputs["last_hidden_state"]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = paddle.split(logits, num_or_sections=1, axis=-1)
        start_logits = start_logits.squeeze(axis=-1)
        end_logits = start_logits.squeeze(axis=-1)
        if return_dict:
            return {
                "start_logits": start_logits,
                "end_logits": end_logits,
                "hidden_states": outputs["all_hidden_states"],
            }
        return start_logits, end_logits
