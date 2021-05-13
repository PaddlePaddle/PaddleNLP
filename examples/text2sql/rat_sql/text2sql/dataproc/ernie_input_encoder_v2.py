#!/usr/bin/env python3
# -*- coding:utf-8 -*-
##########################################################
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved #
##########################################################
"""Encoder input text and database for ernie model

Filname: ernie_input_encoder.py
Authors: ZhangAo(@baidu.com)
Date: 2021-01-25 11:26:46
"""

import sys
import os
import traceback
import logging
import json
from collections import namedtuple
import numpy as np

from ernie import ErnieTokenizer
from paddlenlp.transformers import BertTokenizer

from text2sql.utils import utils
from text2sql.utils import text_utils
from text2sql.dataproc import BaseInputEncoder

ErnieInput = namedtuple(
    "ErnieInput",
    "token_ids sent_ids table_indexes column_indexes value_indexes value_list token_mapping orig_question_tokens candi_nums"
)


class ErnieInputEncoderV2(BaseInputEncoder):
    """使用ernie的文本类型的field_reader，用户不需要自己分词
       处理规则是：自动添加padding,mask,position,task,sentence,并返回length
    """

    padding_id = 0
    truncation_type = 0

    def __init__(self, model_config):
        """
        :param vocab_path:
        """
        super(ErnieInputEncoderV2, self).__init__()

        self.config = model_config
        self.enc_value_with_col = model_config.enc_value_with_col
        if model_config.pretrain_model_type == 'BERT':
            self.tokenizer = BertTokenizer.from_pretrained(
                model_config.pretrain_model)
            self.special_token_dict = {
                # span type
                'table': '[unused1]',
                'column': '[unused2]',
                'value': '[unused3]',
                # column data type
                'text': '[unused11]',
                'real': '[unused12]',
                'number': '[unused13]',
                'time': '[unused14]',
                'binary': '[unused15]',
                'boolean': '[unused16]',
                'bool': '[unused17]',
                'others': '[unused18]',
            }
        else:
            self.tokenizer = ErnieTokenizer.from_pretrained(
                model_config.pretrain_model)
            # 低频token作为特殊标记 ### 其它候选: overchicstoretvhome
            self.special_token_dict = {
                # span type
                'table': 'blogabstract',
                'column': 'wx17house',
                'value': 'fluke62max',
                # column data type
                'text': 'googlemsn',
                'real': 'sputniknews',
                'number': 'sputniknews',
                'time': 'pixstyleme3c',
                'binary': 'pixnetfacebookyahoo',
                'boolean': 'pixnetfacebookyahoo',
                'bool': 'pixnetfacebookyahoo',
                'others': 'ubuntuforumwikilinuxpastechat',
            }
        self._need_bool_value = True if self.config.grammar_type != 'nl2sql' else False

    def check(self, data, db):
        """check

        Args:
            data (TYPE): NULL
            db (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        if len(db.columns) > self.config.max_column_num or len(
                db.tables) > self.config.max_table_num:
            return False
        return True

    def encode(self,
               question,
               db,
               column_match_cells=None,
               candi_nums=None,
               col_orders=None,
               debug=False):
        """encode question and columns
        Arguments:
            question:
            db:
            column_match_cells: 每个 column 下跟 question 匹配度最好的 N 个 cell
            candi_nums: 预测 limit 的候选值。只要不为none，则长度必须大于0（至少包含0、1）
            col_orders:

        Returns: ErnieInput
        """
        question = question.strip()
        if self.config.num_value_col_type != 'q_num':
            orig_question_tokens = text_utils.wordseg(self.question)
            candi_nums = list(
                set(['0', '1'] + text_utils.CandidateValueExtractor.
                    extract_num_from_text(question)))
            candi_nums_index = [-1] * len(candi_nums)
        else:
            orig_question_tokens, candi_nums, candi_nums_index = text_utils.wordseg_and_extract_num(
                question)
            if '0' not in candi_nums:
                candi_nums.append('0')
                candi_nums_index.append(-1)
            if '1' not in candi_nums:
                candi_nums.append('1')
                candi_nums_index.append(-1)
        # schema_indexes = [table_indexes, column_indexes, value_indexes, num_value_indexes]
        tokens, value_list, schema_indexes, token_mapping = \
                self.tokenize(orig_question_tokens, db, column_match_cells, candi_nums, candi_nums_index, col_orders)
        if debug:
            sys.stderr.write(json.dumps(tokens, ensure_ascii=False) + '\n')
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        table_indexes, column_indexes, value_indexes, num_value_indexes = schema_indexes
        q_len = column_indexes[0]
        sent_ids = [0] * q_len + [1] * (len(token_ids) - q_len)

        value_indexes += num_value_indexes
        return ErnieInput(token_ids, sent_ids, table_indexes, column_indexes,
                          value_indexes, value_list, token_mapping,
                          orig_question_tokens, candi_nums)

    def tokenize(self,
                 question,
                 db,
                 column_match_cells=None,
                 candi_nums=None,
                 candi_nums_index=None,
                 col_orders=None):
        """
        Tokenize question and columns and concatenate.

        Parameters:
            question (str): 
            db (Table): 
            col_orders (list or numpy.array): For re-ordering the column columns

        Returns:
            final_tokens: 分3个部分：Question、Schema（含非数字value）、数字value
                          [CLS] Q tokens [SEP]
                          [T] table1 [C] col1 [V] value [C] col2 ... [SEP]
                          [V] number [V] ... [SEP]
            indexes
            token_mapping
        """
        if col_orders is None:
            col_orders = np.arange(len(db.columns))

        if type(question) is str:
            q_tokens_tmp = self.tokenizer.tokenize(question)
            token_idx_mapping = [[i] for i in range(len(q_tokens_tmp))]
        else:
            # question 是经过切词处理的 list
            q_tokens_tmp, token_idx_mapping = self._resplit_words(question)

        final_candi_num_index = []
        if candi_nums_index is not None:
            for idx in candi_nums_index:
                if idx < 0:
                    final_candi_num_index.append(0)
                else:
                    final_candi_num_index.append(token_idx_mapping[idx][0] + 1)

        ## 处理 question tokens
        question_tokens = ['[CLS]'] + q_tokens_tmp
        final_tokens = question_tokens[:self.config.
                                       max_question_len] + ['[SEP]']

        columns = [db.columns[i] for i in col_orders]
        if column_match_cells is not None:
            column_match_cells = [column_match_cells[i] for i in col_orders]
        else:
            column_match_cells = [None] * len(columns)

        ## 处理 schema tokens
        table_indexes = []
        column_indexes = []
        value_indexes = []
        value_list = []
        universe_value_set = set(['是', '否']) if self._need_bool_value else set()
        for idx, (column,
                  match_cells) in enumerate(zip(columns, column_match_cells)):
            if idx == 1 or \
                    idx > 1 and column.table.id != columns[idx - 1].table.id:
                table_indexes.append(len(final_tokens))
                final_tokens.append(self.special_token_dict['table'])
                final_tokens += self.tokenizer.tokenize(column.table.orig_name)

            if idx == 0:
                col_name = '任意列'
                col_type = self.special_token_dict['text']
            else:
                col_name = column.orig_name
                # col_name = remove_brackets(col_name)
                col_type = self.special_token_dict[column.dtype]

            column_indexes.append(len(final_tokens))
            final_tokens += [col_type] + self.tokenizer.tokenize(col_name)

            if match_cells is not None and len(match_cells) > 0:
                if column.dtype in ('text', 'time'):
                    if not self.config.predict_value:
                        match_cells = match_cells[:
                                                  1]  # 模型设置不预测value，则此处仅取1个cell作为column的语义补充
                    for mcell in match_cells:
                        value_list.append(mcell)
                        toks = [self.special_token_dict['value']
                                ] + self.tokenizer.tokenize(mcell)
                        if self.enc_value_with_col:
                            value_indexes.extend(
                                [column_indexes[-1], len(final_tokens)])
                        else:
                            value_indexes.append(len(final_tokens))
                        final_tokens += toks
                elif self.config.predict_value:
                    for mcell in match_cells:
                        universe_value_set.add(mcell)
        final_tokens.append('[SEP]')

        if self.config.predict_value:
            for value in universe_value_set:
                value_list.append(value)
                toks = [self.special_token_dict['value']
                        ] + self.tokenizer.tokenize(value)
                if self.enc_value_with_col:
                    value_indexes.extend([0, len(final_tokens)])
                else:
                    value_indexes.append(len(final_tokens))
                final_tokens += toks
            final_tokens.append('[SEP]')

            ## 处理 number value tokens: 条件 和 limit 的 number values
            num_value_indexes = []
            if candi_nums is not None and len(candi_nums) > 0:
                value_list += candi_nums
                for num, index in zip(candi_nums, final_candi_num_index):
                    if self.enc_value_with_col:
                        # index 是当前 number 在 question 中的下标
                        # len(final_tokens) 是其在整个序列中单独输入时的下标
                        num_value_indexes.extend([index, len(final_tokens)])
                    elif self.config.num_value_col_type == 'q_num':
                        num_value_indexes.append(index)
                    else:
                        num_value_indexes.append(len(final_tokens))
                    final_tokens += [self.special_token_dict['value']
                                     ] + self.tokenizer.tokenize(num)
        else:
            # 不预测value，则固定用特殊的 value/empty 代替
            if self.enc_value_with_col:
                value_indexes = [0, len(final_tokens), 0, len(final_tokens) + 1]
            else:
                value_indexes = [len(final_tokens), len(final_tokens) + 1]
            num_value_indexes = []
            value_list = ['value', 'empty']
            final_tokens.extend(value_list)
        final_tokens.append('[SEP]')

        ###packed_sents_lens = [q_lens, column_tokens_lens, table_tokens_lens, limit_tokens_lens]
        ##packed_sents, packed_sents_lens = self._pack([question_tokens],
        ##                                             column_tokens,
        ##                                             table_tokens,
        ##                                             limit_tokens,
        ##                                             value_indexes=column_values_index)

        return final_tokens, value_list, [
            table_indexes, column_indexes, value_indexes, num_value_indexes
        ], token_idx_mapping

    def _resplit_words(self, words):
        """resplit words by bert_tokenizer

        Args:
            words (TYPE): NULL

        Returns: (list, dict), resplit result and id mapping

        Raises: NULL
        """
        lst_new_result = []
        token_idx_mapping = []
        for idx, word in enumerate(words):
            tokens = self.tokenizer.tokenize(word)
            new_id_start = len(lst_new_result)
            new_id_end = new_id_start + len(tokens)
            lst_new_result.extend(tokens)
            token_idx_mapping.append(list(range(new_id_start, new_id_end)))
        return lst_new_result, token_idx_mapping

    def _pack(self, *sents_of_tokens_list, value_indexes=None):
        packed_sents = []
        packed_sents_lens_all = []
        for sents_of_tokens in sents_of_tokens_list:
            packed_sents_lens = []
            #packed_value_indexes = []
            #for tokens, value_index in zip(sents_of_tokens, [[]] + value_indexes + [[]]):
            for tokens in sents_of_tokens:
                #packed_value_indexes.append(utils.list_increment(value_index, len(packed_sents)))
                packed_tokens = tokens + ['[SEP]']
                packed_sents += packed_tokens
                packed_sents_lens.append(len(packed_tokens))
            packed_sents_lens_all.append(packed_sents_lens)
        return packed_sents, packed_sents_lens_all  #, packed_value_indexes


if __name__ == "__main__":
    """run some simple test cases"""
    if len(sys.argv) != 3:
        print("usage: %s --db db_path")
        sys.exit(1)

    from pathlib import Path
    from text2sql import global_config
    from text2sql.dataproc.dusql_dataset import load_tables

    config = global_config.gen_config()
    parser = ErnieInputEncoderV2(config)
    q = '这 是 一项 测试 。 hello world !'
    db_path = Path(config.data.db)
    db_dict, _ = load_tables(db_path / 'db_schema.json',
                             db_path / 'db_content.json')
    db = db_dict[list(db_dict.keys())[0]]
    column_match_cells = [None] * len(db.columns)
    column_match_cells[1] = ['你好', '[CLS]']
    print(q)
    print([x.orig_name for x in db.columns])
    print([x.orig_name for x in db.tables])
    print(
        parser.encode(
            q,
            db,
            column_match_cells=column_match_cells,
            candi_nums=['1', '0', '10000000'],
            debug=True))
    print('*' * 100)
    print(
        parser.encode(
            q.split(' '), db, candi_nums=['1', '0', '10000000'], debug=True))
