# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import warnings
from collections import OrderedDict

import paddle
import paddle.nn as nn

from .. import PretrainedModel, register_base_model

__all__ = [
    "VisualBertModel",
    "VisualBertForPreTraining",
    "VisualBertForQuestionAnswering",
    "VisualBertForVisualReasoning",
    "VisualBertForMultipleChoice",
]
dtype_float = paddle.get_default_dtype()


class VisualBertEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings and visual embeddings."""
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        visual_embedding_dim=512,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        special_visual_initialize=True,
        pad_token_id=1,
    ):
        super(VisualBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            paddle.arange(max_position_embeddings).expand((1, -1)))

        # For Visual Features
        # Token type and position embedding for image features
        self.visual_token_type_embeddings = nn.Embedding(
            type_vocab_size, hidden_size)
        self.visual_position_embeddings = nn.Embedding(max_position_embeddings,
                                                       hidden_size)

        if special_visual_initialize:
            assert isinstance(self.visual_token_type_embeddings.weight,
                              paddle.Tensor)
            assert isinstance(self.visual_position_embeddings.weight,
                              paddle.Tensor)
            self.visual_token_type_embeddings.weight.set_value(
                self.token_type_embeddings.weight.clone())
            self.visual_position_embeddings.weight.set_value(
                self.position_embeddings.weight.clone())

        self.visual_projection = nn.Linear(visual_embedding_dim, hidden_size)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        # Absolute Position Embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        if visual_embeds is not None:
            if visual_token_type_ids is None:
                visual_token_type_ids = paddle.ones(visual_embeds.shape[:-1],
                                                    dtype=paddle.int64)

            visual_embeds = self.visual_projection(visual_embeds)
            visual_token_type_ids = visual_token_type_ids.astype(paddle.int64)
            visual_token_type_embeddings = self.visual_token_type_embeddings(
                visual_token_type_ids)

            if image_text_alignment is not None:
                # image_text_alignment = Batch x image_length x alignment_number.
                # Each element denotes the position of the word corresponding to the image feature. -1 is the padding value.

                dtype = token_type_embeddings.dtype
                image_text_alignment_mask = (image_text_alignment != -1).long()
                # Get rid of the -1.
                image_text_alignment = image_text_alignment_mask * image_text_alignment

                # Batch x image_length x alignment length x dim
                visual_position_embeddings = self.position_embeddings(
                    image_text_alignment)
                visual_position_embeddings *= image_text_alignment_mask.astype(
                    dtype=dtype).unsqueeze(-1)
                visual_position_embeddings = visual_position_embeddings.sum(2)

                # We want to averge along the alignment_number dimension.
                image_text_alignment_mask = image_text_alignment_mask.astype(
                    dtype=dtype).sum(2)

                if (image_text_alignment_mask == 0).sum() != 0:
                    image_text_alignment_mask[
                        image_text_alignment_mask ==
                        0] = 1  # Avoid divide by zero error
                    warnings.warn(
                        "Found 0 values in `image_text_alignment_mask`. Setting them to 1 to avoid divide-by-zero error."
                    )
                visual_position_embeddings = visual_position_embeddings / image_text_alignment_mask.unsqueeze(
                    -1)

                visual_position_ids = paddle.zeros(*visual_embeds.shape[:-1],
                                                   dtype=paddle.int64)

                # When fine-tuning the detector , the image_text_alignment is sometimes padded too long.
                if visual_position_embeddings.shape[1] != visual_embeds.shape[1]:
                    if visual_position_embeddings.shape[
                            1] < visual_embeds.shape[1]:
                        raise ValueError(
                            f"Visual position embeddings length: {visual_position_embeddings.shape[1]}"
                            f"should be the same as `visual_embeds` length: {visual_embeds.shape[1]}"
                        )
                    visual_position_embeddings = visual_position_embeddings[:, :
                                                                            visual_embeds
                                                                            .
                                                                            shape[
                                                                                1], :]

                visual_position_embeddings = visual_position_embeddings + self.visual_position_embeddings(
                    visual_position_ids)
            else:
                visual_position_ids = paddle.zeros(visual_embeds.shape[:-1],
                                                   dtype=paddle.int64)
                visual_position_embeddings = self.visual_position_embeddings(
                    visual_position_ids)

            visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings

            embeddings = paddle.concat((embeddings, visual_embeddings), axis=1)

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VisualBertEncoder(nn.Layer):
    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1):
        super(VisualBertEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)

        self.layer = nn.TransformerEncoder(encoder_layer, num_hidden_layers)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        """
        Returns:
        last_hidden_state: ``padle.Tensor`` of shape `(batch_size, sequence_length, hidden_size)`
        hidden_states: `optional`, returned when ``output_hidden_states=True`` is passed
        attentions: `optional`, returned when ``output_attentions=True`` is passed
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        assert isinstance(self.layer.layers, nn.LayerList)
        if output_attentions:
            raise NotImplementedError(
                f"nn.TransformerEncoderLayer don't support args: `output_attentions`, Please build an inherit Class to support"
            )

        if head_mask:
            raise NotImplementedError(
                f"nn.TransformerEncoderLayer don't support args: `head_mask`, Please build an inherit Class to support"
            )

        for i, layer_module in enumerate(self.layer.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            # layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, attention_mask)
            # layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions) # retrun a tuple

            hidden_states = layer_outputs
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1], )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                all_hidden_states,
                all_self_attentions,
            ] if v is not None)
        return OrderedDict(last_hidden_state=hidden_states,
                           hidden_states=all_hidden_states,
                           attentions=all_self_attentions)


class VisualBertPooler(nn.Layer):
    def __init__(self, hidden_size=768):
        super(VisualBertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class VisualBertLMPredictionHead(nn.Layer):
    """
    Bert Model with a `language modeling` head on top for CLM fine-tuning.
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(VisualBertLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = self.create_parameter(
            shape=[vocab_size, hidden_size],
            dtype=self.transform.weight.dtype,
            is_bias=False) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = paddle.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias
        return hidden_states
    
    
class VisualBertPreTrainingHeads(nn.Layer):
    """
    Perform language modeling task and next sentence classification task.

    Args:
        hidden_size (int):
            See :class:`BertModel`.
        vocab_size (int):
            See :class:`BertModel`.
        activation (str):
            Activation function used in the language modeling task.
        embedding_weights (Tensor, optional):
            Decoding weights used to map hidden_states to logits of the masked token prediction.
            Its data type should be float32 and its shape is [vocab_size, hidden_size].
            Defaults to `None`, which means use the same weights of the embedding layer.

    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(VisualBertPreTrainingHeads, self).__init__()
        self.predictions = VisualBertLMPredictionHead(hidden_size, vocab_size,
                                                activation, embedding_weights)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        """
        Args:
            sequence_output(Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].
            pooled_output(Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].
            masked_positions(Tensor, optional):
                A tensor indicates positions to be masked in the position embedding.
                Its data type should be int64 and its shape is [batch_size, mask_token_num].
                `mask_token_num` is the number of masked tokens. It should be no bigger than `sequence_length`.
                Defaults to `None`, which means we output hidden-states of all tokens in masked token prediction.

        Returns:
            tuple: Returns tuple (``prediction_scores``, ``seq_relationship_score``).

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].

            - `seq_relationship_score` (Tensor):
                The scores of next sentence prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

        """
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class VisualBertPreTrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    base_model_prefix = "visualbert"
    model_config_file = "model_config.json"

    # pretrained general configuration
    pretrained_init_configuration = {
        "visualbert-vqa": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "visual_embedding_dim": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "bypass_transformer": False,
            "special_visual_initialize": True,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
        },
        "visualbert-vqa-pre": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "visual_embedding_dim": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "bypass_transformer": False,
            "special_visual_initialize": True,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
        },
         "visualbert-vqa-coco-pre": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "visual_embedding_dim": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "bypass_transformer": False,
            "special_visual_initialize": True,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
        },
        "visualbert-nlvr2": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "visual_embedding_dim": 1024,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "bypass_transformer": False,
            "special_visual_initialize": True,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
        },
        "visualbert-nlvr2-pre": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "visual_embedding_dim": 1024,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "bypass_transformer": False,
            "special_visual_initialize": True,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
        },
        "visualbert-nlvr2-coco-pre": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "visual_embedding_dim": 1024,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "bypass_transformer": False,
            "special_visual_initialize": True,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
        },
        "visualbert-vcr": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "visual_embedding_dim": 512,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "bypass_transformer": False,
            "special_visual_initialize": True,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
        },
        "visualbert-vcr-pre": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "visual_embedding_dim": 512,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "bypass_transformer": False,
            "special_visual_initialize": True,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
        },
        "visualbert-vcr-coco-pre": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "visual_embedding_dim": 512,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "bypass_transformer": False,
            "special_visual_initialize": True,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "visualbert-vqa":
            "https://paddlenlp.bj.bcebos.com/models/transformers/visualbert/visualbert-vqa/model_state.pdparams",
            "visualbert-vqa-pre":
            "https://paddlenlp.bj.bcebos.com/models/transformers/visualbert/visualbert-vqa-pre/model_state.pdparams",
            "visualbert-vqa-coco-pre":
            "https://paddlenlp.bj.bcebos.com/models/transformers/visualbert/visualbert-vqa-coco-pre/model_state.pdparams",
            "visualbert-vcr":
            "https://paddlenlp.bj.bcebos.com/models/transformers/visualbert/visualbert-vcr/model_state.pdparams",
            "visualbert-vcr-pre":
            "https://paddlenlp.bj.bcebos.com/models/transformers/visualbert/visualbert-vcr-pre/model_state.pdparams",
            "visualbert-vcr-coco-pre":
            "https://paddlenlp.bj.bcebos.com/models/transformers/visualbert/visualbert-vcr-coco-pre/model_state.pdparams",
            "visualbert-nlvr2":
            "https://paddlenlp.bj.bcebos.com/models/transformers/visualbert/visualbert-nlvr2/model_state.pdparams",
            "visualbert-nlvr2-pre":
            "https://paddlenlp.bj.bcebos.com/models/transformers/visualbert/visualbert-nlvr2-pre/model_state.pdparams",
            "visualbert-nlvr2-coco-pre":
            "https://paddlenlp.bj.bcebos.com/models/transformers/visualbert/visualbert-nlvr2-coco-pre/model_state.pdparams",
        }
    }

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range if hasattr(
                            self, "initializer_range") else
                        self.visual_bert.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12
            # weight 1.0, bias 0.0
        
        # if isinstance(layer, nn.Linear) and layer.bias is not None:
        #     # bias 0.0

    
@register_base_model
class VisualBertModel(VisualBertPreTrainedModel):
    """
    The bare VisualBERT Model transformer outputting raw hidden-states without any specific head on top.
    
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.
    
    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the VisualBERT model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.VisualBertModel`. Vocabulary size of the
            model. Defines the different tokens that can be represented by the ``inputs_ids`` passed to the forward
            method of :class:`~transformers.VisualBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        visual_embedding_dim (:obj:`int`, `optional`, defaults to 512):
            Dimensionality of the visual embeddings to be passed to the model.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling
            :class:`~transformers.VisualBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        bypass_transformer (:obj:bool, optional, defaults to :obj:`False`):
            Whether or not the model should bypass the transformer for the visual embeddings. If set to :obj:`True`,
            the model directly concatenates the visual embeddings from :class:`~transformers.VisualBertEmbeddings` with
            text output from transformers, and then pass it to a self-attention layer.
        special_visual_initialize (:obj:bool, optional, defaults to :obj:`True`):
            Whether or not the visual token type and position type embedding weights should be initialized the same as
            the textual token type and positive type embeddings. When set to :obj:`True`, the weights of the textual
            token type and position type embeddings are copied to the respective visual embedding layers.

    """
    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 visual_embedding_dim=512,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 bypass_transformer=False,
                 special_visual_initialize=True,
                 pad_token_id=1,
                 bos_token_id=0,
                 eos_token_id=2,
                 add_pooling_layer=True):
        super(VisualBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = VisualBertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            visual_embedding_dim=visual_embedding_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            special_visual_initialize=special_visual_initialize,
            pad_token_id=pad_token_id)
        self.encoder = VisualBertEncoder(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob)
        self.pooler = VisualBertPooler(
            hidden_size=hidden_size) if add_pooling_layer else None
        self.bypass_transformer = bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = nn.TransformerEncoderLayer(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                dropout=hidden_dropout_prob,
                activation=hidden_act,
                attn_dropout=attention_probs_dropout_prob,
                act_dropout=0)

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        The VisualBertModel forward method, overrides the `__call__()` special method.
        
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                If its data type is int, the values should be either 0 or 1.

                - **1** for tokens that **not masked**,
                - **0** for tokens that **masked**.

                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            head_mask (Tensor, optional):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (Tensor, optional):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
                vectors than the model's internal embedding lookup matrix.
            visual_embeds (Tensor, optional):
                The embedded representation of the visual inputs, generally derived using using an object detector.
            visual_attention_mask (Tensor, optional):
                Mask to avoid performing attention on visual embeddings. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            visual_token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the visual embeds.
                `What are token type IDs? <../glossary.html#token-type-ids>`_ The authors of VisualBERT set the
                `visual_token_type_ids` to `1` for all tokens.
            image_text_alignment (Tensor, optional):
                Image-Text alignment uses to decide the position IDs of the visual embeddings.
            output_attentions (bool, optional):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
                tensors for more detail.
            output_hidden_states (bool, optional):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
                more detail.
            return_dict (bool, optional):
                Whether or not to return a `OrderedDict` instead of a plain tuple.

        Returns:
            tuple: Returns tuple (`last_hidden_state`, `pooled_output`, `hidden_states` (optional), `attentions` (optional)).
            Or OrderedDict: 
            {
                last_hidden_state=sequence_output,
                pooled_output=pooled_output,
                hidden_states=encoder_outputs['hidden_states'],
                attentions=encoder_outputs['attentions'],
            }
    
            With the fields:

            - `last_hidden_state` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

            - `encoder_outputs` (List(Tensor)):
                A list of Tensor containing hidden-states of the model at each hidden layer in the Transformer encoder.
                The length of the list is `num_hidden_layers`.
                Each Tensor has a data type of float32 and its shape is [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import VisualBertModel, BertTokenizer
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                
                # Built-in pretrained models for VisualBertModel are as follows
                # 'visualbert-vqa-coco-pre'
                # 'visualbert-vqa-pre'
                # 'visualbert-nlvr2-coco-pre'
                # 'visualbert-nlvr2-pre'
                # 'visualbert-vcr-coco-pre'
                # 'visualbert-vcr-pre'
                model = VisualBertModel.from_pretrained('visualbert-vqa-coco-pre')
                
                inputs = tokenizer("Welcome to paddlenlp.", return_attention_mask=True)
                inputs['input_ids'] = paddle.to_tensor([inputs['input_ids']])
                inputs['token_type_ids'] = paddle.to_tensor([inputs['token_type_ids']])
                inputs['attention_mask'] = paddle.to_tensor([inputs['attention_mask']])
                
                # 2048-dim visual embeds for vqa-pre model
                # 1024-dim visual embeds for nlvr2-pre model
                # 512-dim visual embeds for vcr-pre model
                visual_embeds = paddle.ones([100, 2048]).unsqueeze(0) 
                visual_token_type_ids = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)
                visual_attention_mask = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)
                
                return_dict = True
                inputs.update({
                    "visual_embeds": visual_embeds,
                    "visual_token_type_ids": visual_token_type_ids,
                    "visual_attention_mask": visual_attention_mask,
                    "return_dict": return_dict
                })
                
                outputs = model(**inputs)

                if not return_dict:
                    last_hidden_state = outputs[0]
                    pooled_output = outputs[1]
                else:
                    last_hidden_state = outputs['last_hidden_state']
                    pooled_output = outputs['pooled_output']
        """

        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else False)
        return_dict = return_dict if return_dict is not None else False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if visual_embeds is None:
            raise ValueError(
                f"`visual_embeds` can not be of type {type(visual_embeds)} when using a VisualBert Model."
            )

        batch_size, seq_length = input_shape
        # device = input_ids.device if input_ids is not None else inputs_embeds.device

        visual_input_shape = visual_embeds.shape[:-1]

        if attention_mask is None:
            attention_mask = paddle.ones(
                input_shape)  # (batch_size, text_seq_len)

        if visual_attention_mask is None:
            visual_attention_mask = paddle.ones(
                visual_input_shape)  # (batch_size, visual_seq_len)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.

        combined_attention_mask = paddle.concat(
            (attention_mask, visual_attention_mask),
            axis=-1)  # (batch_size, seq_len)
        batch_size, combined_seq_length = combined_attention_mask.shape
        extended_attention_mask = paddle.concat([combined_attention_mask[b].broadcast_to([combined_seq_length, combined_seq_length]).unsqueeze(0).unsqueeze(0) for b in range(batch_size)], axis=0)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
        )
        
        if self.bypass_transformer and visual_embeds is not None:
            text_length = input_ids.shape[1]
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_embedding_output = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[:, :,
                                                                   text_length, :
                                                                   text_length]

            encoded_outputs = self.encoder(
                text_embedding_output,
                attention_mask=text_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoded_outputs[0]
            concatenated_input = paddle.concat(
                (sequence_output, visual_embedding_output), axis=1)
            sequence_output = self.additional_layer(concatenated_input,
                                                    extended_attention_mask)
            pooled_output = self.pooler(
                sequence_output) if self.pooler is not None else None

        else:
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if not return_dict:
                sequence_output = encoder_outputs[0]
            else:
                sequence_output = encoder_outputs['last_hidden_state']

            pooled_output = self.pooler(
                sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return OrderedDict(
            last_hidden_state=sequence_output,
            pooled_output=pooled_output,
            hidden_states=encoder_outputs['hidden_states'],
            attentions=encoder_outputs['attentions'],
        )


class VisualBertForPreTraining(VisualBertPreTrainedModel):
    """
    VisualBert Model with pretraining tasks on top.
    
    Args:
    visual_bert (:class:`VisualBertModel`):
        An instance of :class `VisualBertModel`.
    """
    def __init__(self, visual_bert):
        super(VisualBertForPreTraining, self).__init__()
        self.visual_bert = visual_bert
        self.cls = VisualBertPreTrainingHeads(
            self.visual_bert.config["hidden_size"],
            self.visual_bert.config["vocab_size"],
            self.visual_bert.config["hidden_act"],
            embedding_weights=self.visual_bert.embeddings.word_embeddings.weight)
        
        self.apply(self.init_weights)
    
    def get_predctions_decoder_weight(self):
        return self.cls.predictions.decoder_weight
    
    def get_predctions_decoder_bias(self):
        return self.cls.predictions.decoder_weight
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        sentence_image_labels=None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (Tensor, optional):
                See :class:`VisualBertModel`.
            token_type_ids (Tensor, optional):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                config.max_position_embeddings - 1]``.
            head_mask (Tensor, optional):
                See :class:`VisualBertModel`.
            inputs_embeds (Tensor, optional):
                See :class:`VisualBertModel`.
            visual_embeds (Tensor, optional):
                The embedded representation of the visual inputs, generally derived using using an object detector.
            visual_attention_mask (Tensor, optional):
                See :class:`VisualBertModel`.
            visual_token_type_ids (Tensor, optional):
                See :class:`VisualBertModel`.
            image_text_alignment (Tensor, optional):
                Image-Text alignment uses to decide the position IDs of the visual embeddings.
            output_attentions (bool, optional):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
                tensors for more detail.
            output_hidden_states (bool, optional):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
                more detail.
            return_dict (bool, optional):
                Whether or not to return a `OrderedDict` instead of a plain tuple.
            labels (Tensor, optional):
                Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
                config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
                (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
            sentence_image_labels (Tensor, optional):
                Labels for computing the sentence-image prediction (classification) loss. Input should be a sequence
                pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

                - 0 indicates sequence B is a matching pair of sequence A for the given image,
                - 1 indicates sequence B is a random sequence w.r.t A for the given image.

        Returns:
            
            tuple: Returns tuple (`loss`, `prediction_logits`, `seq_relationship_logits`, `hidden_states`(optional), `attentions`(optional)).
            Or OrderedDict: 
            {
                loss=total_loss,
                prediction_logits=prediction_scores,
                seq_relationship_logits=seq_relationship_score,
                hidden_states=outputs["hidden_states"],
                attentions=outputs["attentions"],
            }
        Example::
            .. code-block::
                import paddle
                from paddlenlp.transformers import VisualBertForPreTraining, BertTokenizer
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

                # Built-in pretrained models for VisualBertModel are as follows
                # 'visualbert-vqa-coco-pre'
                # 'visualbert-vqa-pre'
                # 'visualbert-nlvr2-coco-pre'
                # 'visualbert-nlvr2-pre'
                # 'visualbert-vcr-coco-pre'
                # 'visualbert-vcr-pre'
                model = VisualBertForPreTraining.from_pretrained('visualbert-vqa-coco-pre')
                
                inputs = tokenizer("Welcome to paddlenlp.", return_attention_mask=True)
                inputs['input_ids'] = paddle.to_tensor([inputs['input_ids']])
                inputs['token_type_ids'] = paddle.to_tensor([inputs['token_type_ids']])
                inputs['attention_mask'] = paddle.to_tensor([inputs['attention_mask']])
                
                # 2048-dim visual embeds for vqa-pre model
                # 1024-dim visual embeds for nlvr2-pre model
                # 512-dim visual embeds for vcr-pre model
                visual_embeds = paddle.ones([100, 2048]).unsqueeze(0) 
                visual_token_type_ids = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)
                visual_attention_mask = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)
                
                return_dict = True
                inputs.update({
                    "visual_embeds": visual_embeds,
                    "visual_token_type_ids": visual_token_type_ids,
                    "visual_attention_mask": visual_attention_mask,
                    "return_dict": return_dict
                })
                
                outputs = model(**inputs)

                if not return_dict:
                    loss = outputs[0]
                    prediction_logits = outputs[1]
                    seq_relationship_logits = outputs[2]
                    hidden_states = outputs[3]
                    attentions = outputs[4]
                else:
                    loss = outputs['loss']
                    prediction_logits = outputs['prediction_logits']
                    seq_relationship_logits = outputs['seq_relationship_logits']
                    hidden_states = outputs['hidden_states']
                    attentions = outputs['attentions']
        """
        return_dict = return_dict if return_dict is not None else False

        outputs = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            sequence_output, pooled_output = outputs[:2]
        else:
            sequence_output, pooled_output = outputs['last_hidden_state'], outputs['pooled_output']
            
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and sentence_image_labels is not None:
            total_size = attention_mask.shape[-1] + visual_attention_mask.shape[-1]
            if labels.shape[-1] != total_size:
                raise ValueError(
                    f"The labels provided should have same sequence length as total attention mask."
                    f"Found labels with sequence length {labels.shape[-1]}, expected {total_size}."
                )

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.reshape([-1, self.visual_bert.config["vocab_size"]]), labels.flatten())
            sentence_image_loss = loss_fct(seq_relationship_score.reshape([-1, 2]), sentence_image_labels.flatten())
            total_loss = masked_lm_loss + sentence_image_loss

        if labels is not None and sentence_image_labels is None:
            total_size = attention_mask.shape[-1] + visual_attention_mask.shape[-1]
            if labels.shape[-1] != total_size:
                raise ValueError(
                    f"The labels provided should have same sequence length as total attention mask."
                    f"Found labels with sequence length {labels.shape[-1]}, expected {total_size}."
                )

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            total_loss = loss_fct(prediction_scores.reshape([-1, self.visual_bert.config["vocab_size"]]), labels.flatten())

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return OrderedDict(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs["hidden_states"],
            attentions=outputs["attentions"],
        )


class VisualBertForQuestionAnswering(VisualBertPreTrainedModel):
    """
    VisualBert Model with a classification/regression head on top (a dropout and a linear layer on top of the pooled
    output) for VQA.

    Args:
    visual_bert (:class:`VisualBertModel`):
        An instance of VisualBertModel.
    num_classes (int, optional): 
        The number of classes. Default to `2`.
    dropout (float, optional):
        The dropout probability for output of VisualBERT.
        If None, use the same value as `hidden_dropout_prob` of `VisualBertModel`
        instance `visualbert`. Defaults to `None`.
    """
    def __init__(self, visual_bert, num_classes=2, dropout=None):

        super(VisualBertForQuestionAnswering, self).__init__()
        self.num_classes = num_classes

        self.visual_bert = visual_bert
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  visual_bert.config["hidden_dropout_prob"])
        self.cls = nn.Linear(self.visual_bert.config["hidden_size"],
                             self.num_classes)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        r"""
            labels (Tensor, optional):
                Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
                config.num_labels - 1]`. A KLDivLoss is computed between the labels and the returned logits.
        Returns:
        
        Example::
            .. code-block::
                import paddle
                from paddlenlp.transformers import BertTokenizer, VisualBertForQuestionAnswering
                
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                model = VisualBertForQuestionAnswering.from_pretrained("visualbert-vqa", num_classes=3129)
                model.eval()

                inputs = tokenizer("Who is eating the apple?", return_attention_mask=True)
                inputs['input_ids'] = paddle.to_tensor([inputs['input_ids']])
                inputs['token_type_ids'] = paddle.to_tensor([inputs['token_type_ids']])
                inputs['attention_mask'] = paddle.to_tensor([inputs['attention_mask']])
                visual_embeds = paddle.ones([100,2048]).unsqueeze(0)
                visual_token_type_ids = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)
                visual_attention_mask = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)

                return_dict = True
                inputs.update({
                    "visual_embeds": visual_embeds,
                    "visual_token_type_ids": visual_token_type_ids,
                    "visual_attention_mask": visual_attention_mask,
                    "return_dict": return_dict
                })

                labels = paddle.nn.functional.one_hot(paddle.to_tensor(50), num_classes=3129).astype(paddle.float32) # Batch size 1, Num labels 3092

                with paddle.no_grad():
                    outputs = model(**inputs, labels = labels)

                if not return_dict:
                    loss = outputs[0]
                    logits = outputs[1]
                else:
                    loss = outputs['loss']
                    logits = outputs['logits']
        
        """
        return_dict = return_dict if return_dict is not None else False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        # Get the index of the last text token
        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)

        index_to_gather = attention_mask.sum(1) - 2  # as in original code

        outputs = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        if not return_dict:
            sequence_output = outputs[0]
        else:
            sequence_output = outputs['last_hidden_state']
        # (batch_size, seq_length, hidden_size) 
        # --> gather_index_list 
        # --> (batch_size, seq_length=len(gather_index_list), hidden_size)
        index_to_gather = index_to_gather.astype(paddle.int64)
        pooled_output = paddle.concat([
            paddle.gather(sequence_output[b], index_to_gather[b], axis=0).unsqueeze(0)
            for b in range(input_shape[0])
        ],
                                      axis=0) 
        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        # logits = paddle.transpose(logits, perm=[self.num_classes, 0, 1])
        reshaped_logits = paddle.reshape(logits, shape=[-1, self.num_classes])

        loss = None
        if labels is not None:
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            log_softmax = nn.LogSoftmax(axis=-1)
            reshaped_logits = log_softmax(reshaped_logits)
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return OrderedDict(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs['hidden_states'],
            attentions=outputs['attentions'],
        )


class VisualBertForVisualReasoning(VisualBertPreTrainedModel):
    """
    VisualBert Model with a sequence classification head on top (a dropout and a linear layer on top of the pooled
    output) for Visual Reasoning e.g. for NLVR task.
    
    Args:
    visual_bert (:class:`VisualBertModel`):
        An instance of VisualBertModel.
    num_classes (int, optional): 
        The number of classes. Default to `2`.
    dropout (float, optional):
        The dropout probability for output of VisualBERT.
        If None, use the same value as `hidden_dropout_prob` of `VisualBertModel`
        instance `visualbert`. Defaults to `None`.
    """
    def __init__(self, visual_bert, num_classes=2, dropout=None):
        super(VisualBertForVisualReasoning, self).__init__()
        self.num_classes = num_classes

        self.visual_bert = visual_bert
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  visual_bert.config["hidden_dropout_prob"])
        self.cls = nn.Linear(self.visual_bert.config["hidden_size"],
                             self.num_classes)
        self.apply(self.init_weights)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        r"""
        labels (Tensor, optional):
                Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
                config.num_labels - 1]`. A KLDivLoss is computed between the labels and the returned logits.
        Returns:
        
        Example::
            .. code-block::
                import paddle
                from paddlenlp.transformers import BertTokenizer, VisualBertForVisualReasoning
                
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                model = VisualBertForVisualReasoning.from_pretrained("visualbert-nlvr2", num_classes=2)
                model.eval()

                inputs = tokenizer("Who is eating the apple?", return_attention_mask=True)
                inputs['input_ids'] = paddle.to_tensor([inputs['input_ids']])
                inputs['token_type_ids'] = paddle.to_tensor([inputs['token_type_ids']])
                inputs['attention_mask'] = paddle.to_tensor([inputs['attention_mask']])
                visual_embeds = paddle.ones([100,1024]).unsqueeze(0)
                visual_token_type_ids = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)
                visual_attention_mask = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)

                return_dict = True
                inputs.update({
                    "visual_embeds": visual_embeds,
                    "visual_token_type_ids": visual_token_type_ids,
                    "visual_attention_mask": visual_attention_mask,
                    "return_dict": return_dict
                })

                labels = paddle.to_tensor(1).unsqueeze(0) # Batch size 1, Num choices 2

                with paddle.no_grad():
                    outputs = model(**inputs, labels = labels)

                if not return_dict:
                    loss = outputs[0]
                    logits = outputs[1]
                else:
                    loss = outputs['loss']
                    logits = outputs['logits']
        """
        return_dict = return_dict if return_dict is not None else False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        # Get the index of the last text token
        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)

        outputs = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        if not return_dict:
            pooled_output = outputs[1]
        else:
            pooled_output = outputs['pooled_output']
            
        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        reshaped_logits = paddle.reshape(logits, shape=[-1, self.num_classes])
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.flatten())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return OrderedDict(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs["hidden_states"],
            attentions=outputs["attentions"],
        )

        
class VisualBertForMultipleChoice(VisualBertPreTrainedModel):
    """
    VisualBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for VCR tasks.
    
    Args:
    visual_bert (:class:`VisualBertModel`):
        An instance of VisualBertModel.
    num_classes (int, optional): 
        The number of classes. Default to `1`.
    dropout (float, optional):
        The dropout probability for output of VisualBERT.
        If None, use the same value as `hidden_dropout_prob` of `VisualBertModel`
        instance `visualbert`. Defaults to `None`.
    """
    def __init__(self, visual_bert, num_classes=1, dropout=None):
        super(VisualBertForMultipleChoice, self).__init__()
        self.num_classes = num_classes

        self.visual_bert = visual_bert
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  visual_bert.config["hidden_dropout_prob"])
        self.cls = nn.Linear(self.visual_bert.config["hidden_size"],
                             self.num_classes)
        self.apply(self.init_weights)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        r"""
            labels (Tensor, optional):
                Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
                config.num_labels - 1]`. A KLDivLoss is computed between the labels and the returned logits.
        Returns:
        
        Example::
            .. code-block::
                import paddle
                from paddlenlp.transformers import BertTokenizer, VisualBertForMultipleChoice
        
                text = "Welcome to use paddle paddle and paddlenlp!"
                choice0 = "Use it."
                choice1 = "Like it."
                
                model = VisualBertForMultipleChoice.from_pretrained("visualbert-vcr", num_classes=1)
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                model.eval()

                inputs = tokenizer.batch_encode(batch_text_or_text_pairs=[[text, text], [choice0, choice1]], max_seq_len=128, pad_to_max_seq_len=True, return_attention_mask=True)
                input_ids_list = []
                token_type_ids_list = []
                attention_mask_list = []
                inputs_dict = {}

                for item in inputs: 
                    input_ids_list.append(paddle.to_tensor(item['input_ids']).unsqueeze(0))
                    token_type_ids_list.append(paddle.to_tensor(item['token_type_ids']).unsqueeze(0))
                    attention_mask_list.append(paddle.to_tensor(item['attention_mask']).unsqueeze(0))

                inputs_dict = {
                    "input_ids": paddle.concat(input_ids_list, axis=0).unsqueeze(0),
                    "token_type_ids": paddle.concat(token_type_ids_list, axis=0).unsqueeze(0),
                    "attention_mask": paddle.concat(attention_mask_list, axis=0).unsqueeze(0)
                } 

                visual_embeds = paddle.ones([100,512]).unsqueeze(0)
                visual_embeds = visual_embeds.expand(shape=[1, 2, *visual_embeds.shape])
                visual_token_type_ids = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)
                visual_attention_mask = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)

                labels = paddle.to_tensor(0).unsqueeze(0) # choice0 is correct (according to Wikipedia ;)), batch size 1

                return_dict = True
                inputs_dict.update({
                    "visual_embeds": visual_embeds,
                    "visual_token_type_ids": visual_token_type_ids,
                    "visual_attention_mask": visual_attention_mask,
                    "return_dict": return_dict
                })

                with paddle.no_grad():
                    outputs = model(**inputs_dict, labels=labels)

                if not return_dict:
                    loss = outputs[0]
                    logits = outputs[1]
                else:
                    loss = outputs['loss']
                    logits = outputs['logits']

        """
        return_dict = return_dict if return_dict is not None else False
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.reshape([-1, input_ids.shape[-1]]) if input_ids is not None else None
        attention_mask = attention_mask.reshape([-1, attention_mask.shape[-1]]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape([-1, token_type_ids.shape[-1]]) if token_type_ids is not None else None
        position_ids = position_ids.reshape([-1, position_ids.shape[-1]]) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.reshape([-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1]])
            if inputs_embeds is not None
            else None
        )

        visual_embeds = (
            visual_embeds.reshape([-1, visual_embeds.shape[-2], visual_embeds.shape[-1]])
            if visual_embeds is not None
            else None
        )
        visual_attention_mask = (
            visual_attention_mask.reshape([-1, visual_attention_mask.shape[-1]])
            if visual_attention_mask is not None
            else None
        )
        visual_token_type_ids = (
            visual_token_type_ids.reshape([-1, visual_token_type_ids.shape[-1]])
            if visual_token_type_ids is not None
            else None
        )

        outputs = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            image_text_alignment=image_text_alignment,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            pooled_output = outputs[1]
        else:
            pooled_output = outputs['pooled_output']

        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)
        reshaped_logits = logits.reshape([-1, num_choices])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return OrderedDict(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs["hidden_states"],
            attentions=outputs["attentions"],
        )
