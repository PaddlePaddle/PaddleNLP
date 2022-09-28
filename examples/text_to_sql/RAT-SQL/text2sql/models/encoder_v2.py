#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import traceback
import logging
import json
from pathlib import Path
import attr

import numpy as np
from paddlenlp.transformers import BertModel, ErnieModel, ErniePretrainedModel
import paddle
from paddle import nn
from paddle.nn import functional as F

from text2sql.utils import nn_utils
from text2sql.utils import linking_utils
from text2sql.utils import utils
from text2sql.models import relational_encoder
from text2sql.models import relational_transformer


@attr.s
class EncoderState:
    """Encoder state define"""
    state = attr.ib()
    cls_hidden = attr.ib()
    memory = attr.ib()
    question_memory = attr.ib()
    schema_memory = attr.ib()
    words = attr.ib()

    pointer_memories = attr.ib()
    pointer_maps = attr.ib()

    m2c_align_mat = attr.ib()
    m2t_align_mat = attr.ib()
    m2v_align_mat = attr.ib()

    def find_word_occurrences(self, word):
        """find word occurrences"""
        return [i for i, w in enumerate(self.words) if w == word]


class Text2SQLEncoderV2(nn.Layer):
    """Encoder for text2sql model"""

    def __init__(self, model_config, extra=None):
        super(Text2SQLEncoderV2, self).__init__()
        self.enc_value_with_col = model_config.enc_value_with_col

        self.pretrain_model_type = model_config.pretrain_model_type
        if model_config.pretrain_model_type == 'BERT':
            PretrainModel = BertModel
            vocab_file = os.path.join(
                os.path.expandvars('$HOME'), '.paddlenlp/models',
                model_config.pretrain_model,
                model_config.pretrain_model + '-vocab.txt')
            args = {
                'vocab_size': utils.count_file_lines(vocab_file),
                'type_vocab_size': 2
            }
            self.hidden_size = 768
        elif model_config.pretrain_model_type == 'ERNIE':
            PretrainModel = ErnieModel
            ernie_config = ErniePretrainedModel.pretrained_init_configuration[
                model_config.pretrain_model]
            # with open(Path(model_config.pretrain_model) /
            #           'ernie_config.json') as ifs:
            #     ernie_config = json.load(ifs)
            args = {'cfg': ernie_config}
            self.hidden_size = ernie_config['hidden_size']
        else:
            raise RuntimeError(
                f'unsupported pretrain model type: {model_config.pretrain_model_type}'
            )

        if model_config.init_model_params is None:
            self.base_encoder = PretrainModel.from_pretrained(
                model_config.pretrain_model)
        else:
            self.base_encoder = PretrainModel(**args['cfg'])
        #initializer = nn.initializer.TruncatedNormal(std=0.02)
        self.rel_has_value = True
        self.encs_update = relational_encoder.RelationAwareEncoder(
            num_layers=model_config.rat_layers,
            num_heads=model_config.rat_heads,
            num_relations=len(linking_utils.RELATIONS),
            hidden_size=self.hidden_size,
            has_value=self.rel_has_value)
        if not self.rel_has_value:
            self.value_align = relational_transformer.RelationalPointerNet(
                hidden_size=self.hidden_size, num_relations=0)

        self.include_in_memory = set(['question', 'column', 'table', 'value'])

    def forward(self, inputs):
        """modeling forward stage of encoder
        """
        seq_hidden, cls_hidden = self.base_encoder(inputs['src_ids'],
                                                   inputs['sent_ids'])
        if self.pretrain_model_type != 'ERNIE' and self.pretrain_model_type != 'BERT':
            cls_hidden, seq_hidden = seq_hidden, cls_hidden

        question_tokens_index = inputs["question_tokens_index"]
        table_indexes = inputs["table_indexes"]
        column_indexes = inputs["column_indexes"]
        value_indexes = inputs["value_indexes"]

        question_encs = nn_utils.batch_gather_2d(seq_hidden,
                                                 question_tokens_index)
        table_encs = nn_utils.batch_gather_2d(seq_hidden, table_indexes)
        column_encs = nn_utils.batch_gather_2d(seq_hidden, column_indexes)
        value_encs = nn_utils.batch_gather_2d(seq_hidden, value_indexes)
        if self.enc_value_with_col:
            value_num = value_encs.shape[1] // 2
            value_encs = value_encs.reshape(
                [value_encs.shape[0], value_num, 2, -1]).sum(axis=2)

        orig_inputs = inputs['orig_inputs']
        column_pointer_maps = [{i: [i]
                                for i in range(len(orig_input.columns))}
                               for orig_input in orig_inputs]
        table_pointer_maps = [{i: [i]
                               for i in range(len(orig_input.tables))}
                              for orig_input in orig_inputs]
        value_pointer_maps = [{i: [i]
                               for i in range(len(orig_input.values))}
                              for orig_input in orig_inputs]

        enc_results = []
        # calculate relation encoding one-by-one
        for batch_idx, orig_input in enumerate(orig_inputs):
            q_len = orig_input.column_indexes[0] - 2
            col_size = len(orig_input.columns)
            tab_size = len(orig_input.tables)
            val_size = len(orig_input.values)

            q_enc = question_encs[batch_idx][:q_len]
            tab_enc = table_encs[batch_idx][:tab_size]
            col_enc = column_encs[batch_idx][:col_size]
            val_enc = value_encs[batch_idx][:val_size]

            c_boundary = list(range(col_size + 1))
            t_boundary = list(range(tab_size + 1))

            v_e_input = val_enc.unsqueeze(0) if self.rel_has_value else None
            (q_enc_new, c_enc_new, t_enc_new,
             v_enc_new), align_mat = self.encs_update.forward_unbatched(
                 q_enc.unsqueeze(0), col_enc.unsqueeze(0), tab_enc.unsqueeze(0),
                 c_boundary, t_boundary, orig_input.relations, v_e_input)

            memory = []
            if 'question' in self.include_in_memory:
                memory.append(q_enc_new)
            if 'table' in self.include_in_memory:
                memory.append(t_enc_new)
            if 'column' in self.include_in_memory:
                memory.append(c_enc_new)
            if 'value' in self.include_in_memory and self.rel_has_value:
                memory.append(v_enc_new)
            memory = paddle.concat(memory, axis=1)
            if not self.rel_has_value:
                v_enc_new = val_enc.unsqueeze(0)
                m2v_align_mat = self.value_align(memory,
                                                 v_enc_new,
                                                 relations=None)
                align_mat[2] = m2v_align_mat

            schema_memory = (c_enc_new, t_enc_new)
            if self.rel_has_value:
                schema_memory += (v_enc_new, )

            enc_results.append(
                EncoderState(
                    state=None,
                    cls_hidden=cls_hidden[batch_idx],
                    memory=memory,
                    question_memory=q_enc_new,
                    schema_memory=paddle.concat(schema_memory, axis=1),
                    words=orig_input.question_tokens,
                    pointer_memories={
                        'table': t_enc_new,
                        'column': c_enc_new,
                        'value': v_enc_new,
                    },
                    pointer_maps={
                        'column': column_pointer_maps[batch_idx],
                        'table': table_pointer_maps[batch_idx],
                        'value': value_pointer_maps[batch_idx],
                    },
                    m2c_align_mat=align_mat[0],
                    m2t_align_mat=align_mat[1],
                    m2v_align_mat=align_mat[2],
                ))

        return enc_results

    def span_encoder(self,
                     cls_hidden,
                     seq_hidden,
                     span_index,
                     span_tokens_index,
                     span_tokens_mask,
                     proj_fn=None):
        """encode spans(like headers, table names) by sequence hidden states
        """
        batch_size, max_col_nums, max_col_tokens = span_tokens_index.shape
        hidden_size = cls_hidden.shape[-1]

        # shape = [batch, max_col, hidden_size]
        span_enc1 = nn_utils.batch_gather_2d(seq_hidden, span_index)

        token_gather_index = paddle.reshape(
            span_tokens_index, shape=[-1, max_col_nums * max_col_tokens])
        span_tokens_enc_origin = nn_utils.batch_gather_2d(
            seq_hidden, token_gather_index)

        span_tokens_weight = paddle.reshape(
            paddle.matmul(span_tokens_enc_origin,
                          paddle.unsqueeze(cls_hidden, [-1])),
            [-1, max_col_nums, max_col_tokens])
        span_tokens_weight = F.softmax(nn_utils.sequence_mask(
            span_tokens_weight, span_tokens_mask),
                                       axis=-1)

        span_tokens_enc_origin = paddle.reshape(
            span_tokens_enc_origin,
            [-1, max_col_nums, max_col_tokens, hidden_size])
        span_enc2 = paddle.sum(paddle.multiply(
            span_tokens_enc_origin, span_tokens_weight.unsqueeze([-1])),
                               axis=2)

        span_enc = paddle.concat([span_enc1, span_enc2], axis=-1)
        if proj_fn is not None:
            span_enc = proj_fn(span_enc)
        return span_enc


if __name__ == "__main__":
    """run some simple test cases"""
    inputs = {
        'src_ids':
        paddle.to_tensor(
            np.array([0, 1, 2, 3, 4, 5], dtype=np.int64).reshape([1, 6])),
        'sent_ids':
        paddle.to_tensor(
            np.array([0, 1, 1, 1, 1, 1], dtype=np.int64).reshape([1, 6])),
        'question_tokens_index':
        paddle.to_tensor(list(range(1, 5)), dtype='int64').reshape([1, 4]),
        'column_index':
        paddle.to_tensor([1, 4], dtype='int64').reshape([1, 2]),
        'column_mask':
        paddle.to_tensor([1, 1], dtype='float32').reshape([1, 2]),
        'column_tokens_index':
        paddle.to_tensor([1, 2, 3, 4, 5, 0], dtype='int64').reshape([1, 2, 3]),
        'column_tokens_mask':
        paddle.to_tensor([1, 1, 1, 1, 1, 0], dtype='float32').reshape([1, 2,
                                                                       3]),
        'table_index':
        paddle.to_tensor([1, 4], dtype='int64').reshape([1, 2]),
        'table_mask':
        paddle.to_tensor([1, 1], dtype='float32').reshape([1, 2]),
        'table_tokens_index':
        paddle.to_tensor([1, 2, 3, 4, 5, 0], dtype='int64').reshape([1, 2, 3]),
        'table_tokens_mask':
        paddle.to_tensor([1, 1, 1, 1, 1, 0], dtype='float32').reshape([1, 2,
                                                                       3]),
        'limit_nums_index':
        paddle.to_tensor([1, 4], dtype='int64').reshape([1, 2]),
        'limit_nums_mask':
        paddle.to_tensor([1, 1], dtype='float32').reshape([1, 2]),
        'orig_inputs': [{
            'columns': ['a', 'b'],
            'tables': ['t1', 't2'],
            'question_tokens': ['a', 'bc', 'd'],
            'span_lens': [[6], [1, 1], [1, 1]],
            'relations': np.arange(8 * 8).reshape(8, 8),
        }]
    }

    ##model = Text2SQLEncoder()
    ##outputs = model(inputs)
    ##print(outputs)
