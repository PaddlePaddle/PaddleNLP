# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.fluid.data_feeder import convert_dtype
from paddle import tensor
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder

from ..bert.modeling import BertPooler, BertEmbeddings
from ..model_utils import PretrainedModel, register_base_model

__all__ = [
    'TinyBertModel',
    'TinyBertForPretraining',
    'TinyBertForSequenceClassification',
]


def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


def attention_forward(self,
                      query,
                      key=None,
                      value=None,
                      attn_mask=None,
                      cache=None):
    """
    Redefines the `forward` function of `paddle.nn.MultiHeadAttention`
    """
    key = query if key is None else key
    value = query if value is None else value
    # Compute q ,k ,v
    if cache is None:
        q, k, v = self._prepare_qkv(query, key, value, cache)
    else:
        q, k, v, cache = self._prepare_qkv(query, key, value, cache)

    # Scale dot product attention
    product = tensor.matmul(x=q, y=k, transpose_y=True)

    product *= self.head_dim**-0.5

    if attn_mask is not None:
        # Support bool or int mask
        attn_mask = _convert_attention_mask(attn_mask, product.dtype)
        product = product + attn_mask

    weights = F.softmax(product)
    if self.dropout:
        weights = F.dropout(
            weights,
            self.dropout,
            training=self.training,
            mode="upscale_in_train")

    out = tensor.matmul(weights, v)

    # Combine heads
    out = tensor.transpose(out, perm=[0, 2, 1, 3])
    out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

    # Project to output
    out = self.out_proj(out)

    outs = [out]
    if self.need_weights:
        outs.append(weights)

    # Return attention scores
    outs.append(product)

    if cache is not None:
        outs.append(cache)

    return tuple(outs)


MultiHeadAttention.forward = attention_forward


def transformer_encoder_layer_forward(self, src, src_mask=None, cache=None):
    """
    Redefines the `forward` function of `paddle.nn.TransformerEncoderLayer`
    """
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    residual = src
    if self.normalize_before:
        src = self.norm1(src)
    # Add cache for encoder for the usage like UniLM
    if cache is None:
        src, layer_att = self.self_attn(src, src, src, src_mask)
    else:
        src, layer_att, incremental_cache = self.self_attn(src, src, src,
                                                           src_mask, cache)

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

    return (src, layer_att) if cache is None else (src, layer_att,
                                                   incremental_cache)


TransformerEncoderLayer.forward = transformer_encoder_layer_forward


def transformer_encoder_forward(self, src, src_mask=None, cache=None):
    """
    Redefines the `forward` function of `paddle.nn.TransformerEncoder`
    """
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    output = src
    encoder_layers, encoder_atts, new_caches = [], [], []
    for i, mod in enumerate(self.layers):
        encoder_layers.append(output)
        if cache is None:
            output, layer_att = mod(output, src_mask=src_mask)
        else:
            output, layer_att, new_cache = mod(output,
                                               src_mask=src_mask,
                                               cache=cache[i])
            new_caches.append(new_cache)
        encoder_atts.append(layer_att)

    if self.norm is not None:
        output = self.norm(output)
    encoder_layers.append(output)

    return (encoder_layers, encoder_atts) if cache is None else (
        encoder_layers, encoder_atts, new_caches)


TransformerEncoder.forward = transformer_encoder_forward


class TinyBertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained TinyBERT models. It provides BERT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "tinybert-4l-312d": {
            "vocab_size": 30522,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-4l-312d-v2": {
            "vocab_size": 30522,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d-v2": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "tinybert-4l-312d":
            "http://paddlenlp.bj.bcebos.com/models/transformers/bert/tinybert-4l-312d.pdparams",
            "tinybert-6l-768d":
            "http://paddlenlp.bj.bcebos.com/models/transformers/bert/tinybert-6l-768d.pdparams",
            "tinybert-4l-312d-v2":
            "http://paddlenlp.bj.bcebos.com/models/transformers/bert/tinybert-4l-312d-v2.pdparams",
            "tinybert-6l-768d-v2":
            "http://paddlenlp.bj.bcebos.com/models/transformers/bert/tinybert-6l-768d-v2.pdparams",
        }
    }
    base_model_prefix = "tinybert"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.bert.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class TinyBertModel(TinyBertPretrainedModel):
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0):
        super(TinyBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = BertPooler(hidden_size)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                output_all_encoded_layers=True,
                output_att=True):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, layer_atts = self.encoder(embedding_output,
                                                  attention_mask)

        # "-1" refers to last layer
        pooled_output = self.pooler(encoded_layers[-1])
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if not output_att:
            return encoded_layers, pooled_output

        return encoded_layers, layer_atts, pooled_output


class TinyBertForPretraining(TinyBertPretrainedModel):
    def __init__(self, bert, fit_size=768):
        super(TinyBertForPretraining, self).__init__()
        self.bert = bert
        self.fit_dense = nn.Linear(self.bert.config["hidden_size"], fit_size)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, att_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask)
        tmp = []
        for sequence_layer in sequence_output:
            tmp.append(self.fit_dense(sequence_layer))
        sequence_output = tmp

        return att_output, sequence_output


class TinyBertForSequenceClassification(TinyBertPretrainedModel):
    def __init__(self, bert, num_classes=2, dropout=None, fit_size=768):
        super(TinyBertForSequenceClassification, self).__init__()
        self.bert = bert
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"],
                                    num_classes)
        self.fit_dense = nn.Linear(self.bert.config["hidden_size"], fit_size)
        self.activation = nn.ReLU()
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                is_student=False):

        sequence_output, att_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=True,
            output_att=True)

        logits = self.classifier(self.activation(pooled_output))

        tmp = []
        if is_student:
            for sequence_layer in sequence_output:
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp

        return logits, att_output, sequence_output
