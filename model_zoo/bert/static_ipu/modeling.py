# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import logging

import numpy as np
import paddle
import paddle.nn as nn
import paddle.static
import paddle.fluid
from paddle.nn import Layer
from typing import List, NamedTuple, Optional
from contextlib import ExitStack


class DeviceScope(object):

    def __init__(self, index, stage, name_scope=None):
        self.index = index
        self.stage = stage
        self.name_scope = name_scope

    def __enter__(self):
        self.stack = ExitStack()
        self.stack.enter_context(
            paddle.static.ipu_shard_guard(index=self.index, stage=self.stage))
        if self.name_scope is not None:
            self.stack.enter_context(paddle.static.name_scope(self.name_scope))
        return self

    def __exit__(self, *exp):
        self.stack.close()
        return False


class IpuBertConfig(NamedTuple):
    """
    The configuration for BERT Model.
    Args:
        seq_len (int):
            The sequence length. Default to `128`.
        max_position_embeddings (int):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        max_predictions_per_seq (int):
            The max number of the masked token each sentence. Default to `20`.
        hidden_size (int):
            Dimensionality of the embedding layer, encoder layer and pooler layer. Defaults to `768`.
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `BertModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `BertModel`.
        num_hidden_layers (int):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        available_mem_proportion (float):
            The available proportion of memory used by conv or matmul. Default to `0.28`.
        type_vocab_size (int):
            The vocabulary size of `token_type_ids`.
            Defaults to `2`.
        hidden_dropout_prob (float):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        task (str):
            The type of the NLP model.
        layers_per_ipu (list):
            Number of attention layers executed on each IPU.
    """
    micro_batch_size: int = 1
    seq_len: int = 128
    max_position_embeddings: int = 512
    max_predictions_per_seq: int = 20
    hidden_size: int = 768
    vocab_size: int = 30400
    num_hidden_layers: int = 12
    available_mem_proportion: float = 0.28
    type_vocab_size: int = 2

    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # Choices: PRETRAINING (MLM + NSP), SQUAD
    task: str = "PRETRAINING"
    layers_per_ipu: List = None

    embeddings_scope: DeviceScope = None
    attn_scopes: DeviceScope = None
    ff_scopes: DeviceScope = None
    mlm_scope: DeviceScope = None
    nsp_scope: DeviceScope = None


class IpuBertEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config, custom_ops=None):
        super(IpuBertEmbeddings, self).__init__()
        self.config = config
        self.word_embeddings_weights = self.create_parameter(
            shape=[config.hidden_size, config.vocab_size], dtype="float32")
        self.token_embeddings_weights = self.create_parameter(
            shape=[config.type_vocab_size, config.hidden_size], dtype="float32")
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=0.001)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.custom_ops = custom_ops

    def forward(self, indices, segments, positions):
        # word embeddings
        word_embeddings_weights = paddle.transpose(self.word_embeddings_weights,
                                                   [1, 0])
        input_embeddings = paddle.gather(word_embeddings_weights,
                                         indices,
                                         axis=0)

        # position_embeddings
        position_embeddings = self.position_embeddings(positions)

        # token_type_embeddings
        token_type_embeddings = paddle.fluid.input.one_hot(segments, depth=2)
        token_type_embeddings = paddle.matmul(token_type_embeddings,
                                              self.token_embeddings_weights)

        embeddings = paddle.add(input_embeddings, position_embeddings)
        embeddings = paddle.add(embeddings, token_type_embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, self.word_embeddings_weights


class BertModel(Layer):
    """
    The bare BERT Model transformer outputting raw hidden-states.

    This model refers to :class:`~paddlenlp.transformers.bert.BertModel`.

    Args:
        config (IpuBertConfig):
            configuration of bert.
        custom_ops:
            custom defined operators which can be found in directory `custom_ops`.
    """

    def __init__(self, config, custom_ops=None):
        super(BertModel, self).__init__()
        self.config = config
        self.custom_ops = custom_ops

        qk_scale = 1 / np.sqrt(
            self.config.hidden_size / self.config.num_hidden_layers)
        self.qk_scale_attrs = {
            'name': 'QK_scale',
            'shape': [1],
            'dtype': 'float32',
            'value': qk_scale,
        }
        self.qkv_shape = [-1, self.config.seq_len, 12, 64]
        self.masks = {}

        self.embedding = IpuBertEmbeddings(self.config, custom_ops)

    def _encoder_layer_ipu_offset(self, layer_index):
        encoder_index = 0
        if len(self.config.layers_per_ipu) == 1:
            encoder_index = layer_index // self.config.layers_per_ipu[0]
        else:
            for ipu, num_layers in enumerate(self.config.layers_per_ipu):
                layer_index -= num_layers
                if layer_index < 0:
                    encoder_index = ipu
                    break
        return encoder_index

    def should_checkpoint(self, layer_index):
        encoder_index = self._encoder_layer_ipu_offset(layer_index)
        if len(self.config.layers_per_ipu) == 1:
            layers = self.config.layers_per_ipu[0]
            layer_index -= encoder_index * layers
        else:
            layers = self.config.layers_per_ipu[encoder_index]
            layer_index -= sum(self.config.layers_per_ipu[:encoder_index])
        return layer_index < (layers - 1)

    def forward(self, indices, segments, positions, input_mask):
        r'''
        The BertModel forward method, overrides the `__call__()` special method.

        Args:
            indices (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int32` and it has a shape of [batch_size * sequence_length].
            segments (Tensor):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                Its data type should be `int32` and it has a shape of [batch_size * sequence_length].
            positions(Tensor):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `[batch_size * sequence_length]` and dtype as int32.
            input_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                If the task is PRETRAINING:
                    input_mask[0] is the index that masking starts in the mask_tokens
                    input_mask[1] is the index that masking starts in the rest of the sequence
                Otherwise
                    input_mask is the mask tensor that has -1000 in positions to be masked and 0 otherwise.

        Returns:
            tuple: Returns tuple (`sequence_output`, `word_embeddings_weights`).

            With the fields:

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].
        '''

        with self.config.embeddings_scope:
            sequence_output, word_embeddings_weights = self.embedding(
                indices, segments, positions)

        if self.config.task == "PRETRAINING":
            with paddle.static.ipu_shard_guard(index=0, stage=0):
                input_mask[0] = self.custom_ops.detach(input_mask[0])
                input_mask[1] = self.custom_ops.detach(input_mask[1])

        for i in range(self.config.num_hidden_layers):
            # Attention
            attn_scope = self.config.attn_scopes[i]
            with attn_scope:
                with paddle.static.name_scope(f"Layer{i}/Attention"):
                    layer_input = sequence_output
                    q = self.create_parameter(shape=[
                        self.config.hidden_size, self.config.hidden_size
                    ],
                                              dtype="float32")
                    k = self.create_parameter(shape=[
                        self.config.hidden_size, self.config.hidden_size
                    ],
                                              dtype="float32")
                    v = self.create_parameter(shape=[
                        self.config.hidden_size, self.config.hidden_size
                    ],
                                              dtype="float32")
                    qkv = paddle.concat([q, k, v], axis=1)
                    qkv = paddle.matmul(sequence_output, qkv)
                    qkv.block.ops[-1]._set_attr(
                        '__available_memory',
                        self.config.available_mem_proportion)
                    q, k, v = paddle.split(qkv,
                                           num_or_sections=[
                                               self.config.hidden_size,
                                               self.config.hidden_size,
                                               self.config.hidden_size
                                           ],
                                           axis=1)
                    q = paddle.reshape(q, self.qkv_shape)
                    q = paddle.transpose(q, [0, 2, 1, 3])
                    k = paddle.reshape(k, self.qkv_shape)
                    k = paddle.transpose(k, [0, 2, 3, 1])
                    v = paddle.reshape(v, self.qkv_shape)
                    v = paddle.transpose(v, [0, 2, 1, 3])

                    # Attention calculation
                    with paddle.static.name_scope(f"Z"):
                        if self.config.task == "PRETRAINING":
                            if attn_scope.index in self.masks:
                                final_mask = self.masks[attn_scope.index]
                            else:
                                with paddle.static.name_scope("Mask"):
                                    base_value = np.arange(
                                        self.config.seq_len).astype('int32')
                                    base = paddle.fluid.layers.assign(
                                        base_value)
                                    mmask = paddle.less_than(
                                        base, input_mask[0])
                                    mask_value = np.greater_equal(
                                        base_value,
                                        self.config.max_predictions_per_seq)
                                    mask = paddle.fluid.layers.assign(
                                        mask_value)
                                    mmask = paddle.logical_or(mmask, mask)
                                    smask = paddle.less_than(
                                        base, input_mask[1])
                                    final_mask = paddle.logical_and(
                                        mmask, smask)
                                    final_mask = paddle.cast(
                                        final_mask, "float16")
                                    sub_attrs = {
                                        'name': 'constant_sub',
                                        'shape': [1],
                                        'dtype': 'float32',
                                        'value': 1,
                                    }
                                    mul_attrs = {
                                        'name': 'constant_mul',
                                        'shape': [1],
                                        'dtype': 'float32',
                                        'value': 1000,
                                    }
                                    final_mask = paddle.fluid.layers.elementwise_sub(
                                        final_mask,
                                        paddle.fluid.layers.fill_constant(
                                            **sub_attrs))
                                    final_mask = paddle.fluid.layers.elementwise_mul(
                                        final_mask,
                                        paddle.fluid.layers.fill_constant(
                                            **mul_attrs))
                                    final_mask = paddle.reshape(
                                        final_mask,
                                        [-1, 1, 1, self.config.seq_len])
                                    final_mask = self.custom_ops.detach(
                                        final_mask)
                                    self.masks[attn_scope.index] = final_mask

                        qk = paddle.matmul(q, k)
                        qk.block.ops[-1]._set_attr(
                            '__available_memory',
                            self.config.available_mem_proportion)
                        qk_scale = paddle.fluid.layers.fill_constant(
                            **self.qk_scale_attrs)
                        qk = paddle.fluid.layers.elementwise_mul(qk, qk_scale)

                        if self.config.task == "PRETRAINING":
                            qk = paddle.fluid.layers.elementwise_add(
                                qk, final_mask)
                        else:
                            # for SQUAD task, input_mask is calculated in data preprocessing
                            qk = paddle.fluid.layers.elementwise_add(
                                qk, input_mask)

                        qk = paddle.fluid.layers.softmax(qk)
                        if self.config.task == "SQUAD":
                            qk = paddle.fluid.layers.dropout(
                                qk,
                                self.config.attention_probs_dropout_prob,
                                dropout_implementation='upscale_in_train')
                        qkv = paddle.matmul(qk, v)
                        qkv.block.ops[-1]._set_attr(
                            '__available_memory',
                            self.config.available_mem_proportion)
                        qkv = paddle.transpose(qkv, [0, 2, 1, 3])
                        qkv = paddle.reshape(qkv, [-1, self.config.hidden_size])

                    qkv_linear = nn.Linear(self.config.hidden_size,
                                           self.config.hidden_size,
                                           bias_attr=False)
                    qkv = qkv_linear(qkv)
                    qkv.block.ops[-1]._set_attr(
                        '__available_memory',
                        self.config.available_mem_proportion)
                    qkv = paddle.fluid.layers.dropout(
                        qkv,
                        self.config.attention_probs_dropout_prob,
                        dropout_implementation='upscale_in_train')
                    attention = paddle.add(layer_input, qkv)
                    layer_norm1 = nn.LayerNorm(self.config.hidden_size,
                                               epsilon=0.001)
                    attention = layer_norm1(attention)

            # FF
            with self.config.ff_scopes[i]:
                with paddle.static.name_scope(f"Layer{i}/FF"):
                    ff_linear1 = nn.Linear(self.config.hidden_size,
                                           4 * self.config.hidden_size)
                    ff_linear2 = nn.Linear(4 * self.config.hidden_size,
                                           self.config.hidden_size)
                    with paddle.static.name_scope(f"1"):
                        ff = ff_linear1(attention)
                        ff.block.ops[-2]._set_attr(
                            '__available_memory',
                            self.config.available_mem_proportion)
                    ff = paddle.fluid.layers.gelu(ff, approximate=True)
                    with paddle.static.name_scope(f"2"):
                        ff = ff_linear2(ff)
                        ff.block.ops[-2]._set_attr(
                            '__available_memory',
                            self.config.available_mem_proportion)
                    ff = paddle.fluid.layers.dropout(
                        ff,
                        self.config.attention_probs_dropout_prob,
                        dropout_implementation='upscale_in_train')
                    ff = paddle.add(attention, ff)
                    layer_norm2 = nn.LayerNorm(self.config.hidden_size,
                                               epsilon=0.001)
                    sequence_output = layer_norm2(ff)

                if self.should_checkpoint(i):
                    with paddle.static.name_scope(f"Layer{i}"):
                        logging.info(f'add checkpointoutput for ff_{i}')
                        sequence_output = self.custom_ops.checkpointoutput(
                            sequence_output)
        return sequence_output, word_embeddings_weights


class IpuBertForQuestionAnswering(Layer):
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and
    `span end logits`).

    Args:
        hidden_size (int):
            Dimensionality of the embedding layer, encoder layer and pooler layer. Defaults to `768`.
        seq_len (int):
            See :class:`IpuBertConfig`.
        """

    def __init__(self, hidden_size, seq_len):
        super(IpuBertForQuestionAnswering, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output):
        r"""
        The IpuBertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            sequence_output (Tensor):
                See :class:`BertModel`.

        Returns:
            tuple: Returns tuple (`start_logits`, `end_logits`).

            With the fields:

            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].
        """
        logits = self.classifier(sequence_output)

        start_logits = paddle.slice(input=logits,
                                    axes=[1],
                                    starts=[0],
                                    ends=[1])
        end_logits = paddle.slice(input=logits, axes=[1], starts=[1], ends=[2])

        start_logits = paddle.reshape(start_logits, [-1, self.seq_len])
        end_logits = paddle.reshape(end_logits, [-1, self.seq_len])
        return start_logits, end_logits


class IpuBertQAAccAndLoss(paddle.nn.Layer):
    """
    Criterion for Question and Answering.
    """

    def __init__(self, custom_ops=None):
        super(IpuBertQAAccAndLoss, self).__init__()
        self.custom_ops = custom_ops

    def forward(self, start_logits, end_logits, start_labels, end_labels):
        r"""
        The IpuBertQAAccAndLoss forward method, overrides the __call__() special method.

        Args:
            start_logits (Tensor):
                See :class:`IpuBertForQuestionAnswering`.
            end_logits (Tensor):
                See :class:`IpuBertForQuestionAnswering`.
            start_labels (Tensor):
                Labels for start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].
            end_labels (Tensor):
                Labels for end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

        """
        with paddle.static.name_scope("loss"):
            start_loss = paddle.fluid.layers.softmax(start_logits)
            start_loss = self.custom_ops.custom_nll_loss(
                start_loss, start_labels, 1, "None", False)
            end_loss = paddle.fluid.layers.softmax(end_logits)
            end_loss = self.custom_ops.custom_nll_loss(end_loss, end_labels, 1,
                                                       "None", False)
            loss = paddle.add(start_loss, end_loss)

        with paddle.static.name_scope("acc"):
            start_logits = paddle.fluid.layers.argmax(start_logits, axis=1)
            end_logits = paddle.fluid.layers.argmax(end_logits, axis=1)
            start_equal = paddle.fluid.layers.equal(start_logits, start_labels)
            end_equal = paddle.fluid.layers.equal(end_logits, end_labels)
            start_equal = paddle.fluid.layers.cast(start_equal, 'float32')
            end_equal = paddle.fluid.layers.cast(end_equal, 'float32')
            start_acc = paddle.mean(start_equal)
            end_acc = paddle.mean(end_equal)

        return start_acc, end_acc, loss


class IpuBertPretrainingMLMHeads(Layer):
    """
    Perform language modeling task.

    Args:
        hidden_size (int):
            See :class:`IpuBertConfig`.
        vocab_size (int):
            See :class:`IpuBertConfig`.
        max_position_embeddings (int):
            See :class:`IpuBertConfig`.
        max_predictions_per_seq (int):
            See :class:`IpuBertConfig`.
        seq_len (int):
            See :class:`IpuBertConfig`.
    """

    def __init__(self, hidden_size, vocab_size, max_position_embeddings,
                 max_predictions_per_seq, seq_len):
        super(IpuBertPretrainingMLMHeads, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.max_predictions_per_seq = max_predictions_per_seq
        self.sequence_length = seq_len
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=0.001)

    def forward(self, encoders_output, word_embeddings_weights):
        # cls
        out = self.transform(encoders_output)
        out = paddle.fluid.layers.gelu(out, approximate=True)
        out = self.layer_norm(out)

        # mlm
        out = paddle.reshape(out, [-1, self.sequence_length, self.hidden_size])
        out = paddle.slice(out, [1], [0], [self.max_predictions_per_seq])
        out = paddle.reshape(out, [-1, self.hidden_size])

        # serialized matmul
        out = paddle.matmul(out, word_embeddings_weights)
        out.block.ops[-1]._set_attr('serialize_factor', 5)
        mlm_out = paddle.reshape(
            out, [-1, self.max_predictions_per_seq, self.vocab_size])

        return mlm_out


class IpuBertPretrainingNSPHeads(Layer):
    """
    Perform next sequence classification task.

    Args:
        hidden_size (int):
            See :class:`IpuBertConfig`.
        max_predictions_per_seq (int):
            See :class:`IpuBertConfig`.
        seq_len (int):
            See :class:`IpuBertConfig`.
    """

    def __init__(self, hidden_size, max_predictions_per_seq, seq_len):
        super(IpuBertPretrainingNSPHeads, self).__init__()
        self.hidden_size = hidden_size
        self.max_predictions_per_seq = max_predictions_per_seq
        self.seq_len = seq_len
        self.seq_relationship = nn.Linear(hidden_size, 2)
        self.pooler = IpuBertPooler(hidden_size, self.seq_len,
                                    self.max_predictions_per_seq)

    def forward(self, encoders_output):
        pooled_output = self.pooler(encoders_output)
        nsp_out = self.seq_relationship(pooled_output)
        return nsp_out


class IpuBertPooler(Layer):
    """
    Pool the result of BertEncoder.
    """

    def __init__(self,
                 hidden_size,
                 sequence_length,
                 max_predictions_per_seq,
                 pool_act="tanh"):
        super(IpuBertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = pool_act
        self.sequence_length = sequence_length
        self.max_predictions_per_seq = max_predictions_per_seq
        self.hidden_size = hidden_size

    def forward(self, hidden_states):
        hidden_states = paddle.reshape(
            hidden_states, [-1, self.sequence_length, self.hidden_size])
        first_token_tensor = paddle.slice(
            input=hidden_states,
            axes=[1],
            starts=[self.max_predictions_per_seq],
            ends=[self.max_predictions_per_seq + 1])
        first_token_tensor = paddle.reshape(first_token_tensor,
                                            [-1, self.hidden_size])
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class IpuBertPretrainingMLMAccAndLoss(Layer):
    """
    Criterion for masked language modeling.
    """

    def __init__(self, micro_batch, ignore_index, custom_ops):
        super(IpuBertPretrainingMLMAccAndLoss, self).__init__()
        self.micro_batch = micro_batch
        self.ignore_index = ignore_index
        self.custom_ops = custom_ops

    def forward(self, mlm, masked_lm_ids):
        mlm_pred = paddle.fluid.layers.argmax(mlm, axis=-1)
        mlm_pred = paddle.cast(mlm_pred, "int32")
        with paddle.static.name_scope("Accuracy"):
            mlm_label = paddle.cast(masked_lm_ids, "int32")
            mlm_correct = paddle.fluid.layers.equal(mlm_pred, mlm_label)
            attrs = {
                'name': 'mlm_mask_val',
                'shape': [1],
                'dtype': 'int32',
                'value': self.ignore_index,
            }
            mlm_mask_val = paddle.fluid.layers.fill_constant(**attrs)
            mlm_unmask = paddle.fluid.layers.equal(mlm_label, mlm_mask_val)
            mlm_mask = paddle.logical_not(mlm_unmask)
            mlm_mask = paddle.cast(mlm_mask, "float32")
            mlm_correct = paddle.cast(mlm_correct, "float32")
            masked_mlm_correct = paddle.fluid.layers.elementwise_mul(
                mlm_correct, mlm_mask)
            total_correct_tokens = paddle.fluid.layers.reduce_sum(
                masked_mlm_correct)
            total_tokens = paddle.fluid.layers.reduce_sum(mlm_mask)
            total_correct_tokens = paddle.cast(total_correct_tokens, "float32")
            total_tokens = paddle.cast(total_tokens, "float32")
            mlm_acc = paddle.fluid.layers.elementwise_div(
                total_correct_tokens, total_tokens)

        masked_lm_softmax = paddle.fluid.layers.softmax(mlm)
        mlm_loss = self.custom_ops.custom_nll_loss(masked_lm_softmax,
                                                   masked_lm_ids, 1,
                                                   str(self.ignore_index),
                                                   False)

        return mlm_acc, mlm_loss


class IpuBertPretrainingNSPAccAndLoss(Layer):
    """
    Criterion for next sequence classification.
    """

    def __init__(self, micro_batch, ignore_index, custom_ops):
        super(IpuBertPretrainingNSPAccAndLoss, self).__init__()
        self.micro_batch = micro_batch
        self.ignore_index = ignore_index
        self.custom_ops = custom_ops

    def forward(self, nsp, nsp_label):
        nsp_pred = paddle.fluid.layers.argmax(nsp, axis=-1)
        nsp_pred = paddle.cast(nsp_pred, "int32")
        with paddle.static.name_scope("Accuracy"):
            nsp_label = paddle.cast(nsp_label, "int32")
            nsp_correct = paddle.fluid.layers.equal(nsp_pred, nsp_label)
            nsp_correct = paddle.cast(nsp_correct, "int32")
            nsp_correct = paddle.fluid.layers.reduce_sum(nsp_correct)
            nsp_correct = paddle.cast(nsp_correct, "float32")
            attrs = {
                'name': 'mlm_mask_val',
                'shape': [1],
                'dtype': 'int32',
                'value': self.micro_batch,
            }
            nsp_total = paddle.fluid.layers.fill_constant(**attrs)
            nsp_total = paddle.cast(nsp_total, "float32")
            nsp_acc = paddle.fluid.layers.elementwise_div(
                nsp_correct, nsp_total)

        next_sentence_softmax = paddle.fluid.layers.softmax(nsp)
        nsp_loss = self.custom_ops.custom_nll_loss(next_sentence_softmax,
                                                   nsp_label, 1, "None", False)

        return nsp_acc, nsp_loss
