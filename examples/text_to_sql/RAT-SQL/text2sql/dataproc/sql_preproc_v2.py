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
import collections
import collections.abc
import copy
import itertools
import shutil
from pathlib import Path
import attr

import numpy as np
import paddle
import paddle.nn.functional as F

from text2sql.dataproc import vocab
from text2sql.utils import serialization


def get_field_presence_info(ast_wrapper, node, field_infos):
    """get_field_presence_info"""
    present = []
    for field_info in field_infos:
        field_value = node.get(field_info.name)
        is_present = field_value is not None and field_value != []

        maybe_missing = field_info.opt or field_info.seq
        is_builtin_type = field_info.type in ast_wrapper.primitive_types

        if maybe_missing and is_builtin_type:
            # TODO: make it possible to deal with "singleton?"
            present.append(is_present and type(field_value).__name__)
        elif maybe_missing and not is_builtin_type:
            present.append(is_present)
        elif not maybe_missing and is_builtin_type:
            present.append(type(field_value).__name__)
        elif not maybe_missing and not is_builtin_type:
            assert is_present
            present.append(True)
    return tuple(present)


@attr.s
class DecoderSQLItem:
    """DecoderSQLItem"""
    tree = attr.ib()
    orig_code = attr.ib()
    sql_query = attr.ib(default="")


class SQLPreproc(object):
    """SQLPreproc"""

    def __init__(self,
                 base_path,
                 grammar_class,
                 predict_value=True,
                 min_freq=3,
                 max_count=5000,
                 use_seq_elem_rules=False,
                 is_cached=False):
        """
        Args:
            base_path (TYPE): if is_cached is False, base_path is the asdl grammar file.
                              if is_cached is True, base_path is path to cached directory.
            grammar_class (TYPE): grammar class, like grammars.dusql.DuSQLLanguage
            predict_value (TYPE): Default is True
            min_freq (TYPE): Default is 3
            max_count (TYPE): Default is 5000
            use_seq_elem_rules (TYPE): Default is False
            is_cached (TYPE): Default is False

        Raises: NULL
        """
        self.base_path = base_path
        self.predict_value = predict_value
        self.vocab = None
        self.all_rules = None
        self.rules_mask = None

        # key: train/dev/val/test/...
        # value: examples
        self.items = collections.defaultdict(list)
        self.sum_type_constructors = collections.defaultdict(set)
        self.field_presence_infos = collections.defaultdict(set)
        self.seq_lengths = collections.defaultdict(set)
        self.primitive_types = set()

        if not is_cached:
            self.grammar = grammar_class(self.base_path)
            self.ast_wrapper = self.grammar.ast_wrapper
            self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
        else:
            self.grammar = None
            self.ast_wrapper = None
            self.load(grammar_class)

        self.use_seq_elem_rules = use_seq_elem_rules
        if self.predict_value:
            self.format_sql_value = self.transfer_sql_value
        else:
            self.format_sql_value = self.fix_sql_value

    def _get_val_index(self, val, value_dict):

        def _float(val):
            try:
                return True, str(int(float(val)))
            except Exception as e:
                return False, ''

        val = str(val)
        if val in value_dict:
            return value_dict[val]
        is_float, new_val = _float(val)
        if is_float and new_val in value_dict:
            return value_dict[new_val]

        new_val = val.replace('.', '')
        candi = []
        for v, idx in value_dict.items():
            v = v.replace('.', '')
            if v.startswith(new_val) or new_val.startswith(v):
                candi.append((v, idx))

        if len(candi) == 1:
            return candi[0][1]
        elif len(candi) > 1:
            candi.sort(key=lambda x: len(x[0]), reverse=True)
            return candi[0][1]

        return -1

    def transfer_sql_value(self, sql_json, value_dict):
        """transfer value str to int index
        """
        if 'cond_conn_op' in sql_json:
            self.transfer_simple_sql_value(sql_json, value_dict)
            return

        def _trans_cond(cond):
            """transfer condition value"""
            val1 = cond[3]
            val2 = cond[4]
            if type(val1) is dict:
                self.transfer_sql_value(val1, value_dict)
                if val2 is not None:
                    val2 = self._get_val_index(val2, value_dict)
                    cond[4] = val2 if val2 >= 0 else 0
                return

            val1 = self._get_val_index(val1, value_dict)
            if val2 is not None:
                val2 = self._get_val_index(val2, value_dict)
            if val1 == -1:
                val1 = 0
                logging.debug('lost value: %s. candidates: %s', cond[3],
                              ', '.join(value_dict.keys()))
                logging.debug('sql is: %s',
                              json.dumps(sql_json, ensure_ascii=False))
            if val2 == -1:
                val2 = 0
            cond[3] = val1
            cond[4] = val2

        for table_unit in sql_json['from']['table_units']:
            if type(table_unit[1]) is dict:
                self.transfer_sql_value(table_unit[1], value_dict)

        for cond in sql_json['where'][::2]:
            _trans_cond(cond)
        for cond in sql_json['having'][::2]:
            _trans_cond(cond)

        if sql_json['limit'] is not None:
            limit = str(sql_json['limit'])
        else:
            limit = '0'
        if limit in value_dict:
            sql_json['limit'] = value_dict[limit]
        else:
            logging.debug('value of limit is lost: %s. candidates: %s', limit,
                          ', '.join(value_dict.keys()))
            sql_json['limit'] = value_dict['0']

        if sql_json['intersect'] is not None:
            self.transfer_sql_value(sql_json['intersect'], value_dict)
        if sql_json['union'] is not None:
            self.transfer_sql_value(sql_json['union'], value_dict)
        if sql_json['except'] is not None:
            self.transfer_sql_value(sql_json['except'], value_dict)

    def transfer_simple_sql_value(self, sql_json, value_dict):
        for cond in sql_json['conds']:
            value = cond[2]
            new_val = self._get_val_index(value, value_dict)
            if new_val == -1:
                new_val = 0
            cond[2] = new_val

    def fix_sql_value(self, sql_json, value_dict):
        """fix sql value to 'value' token
        """

        def _fix_cond_value(cond):
            """transfer condition value"""
            val1 = cond[3]
            val2 = cond[4]
            if type(val1) is dict:
                self.fix_sql_value(val1, value_dict)
                if val2 is not None:
                    val2 = self._get_val_index('value', value_dict)
                    cond[4] = val2 if val2 >= 0 else 0
                return

            val1 = self._get_val_index('value', value_dict)
            if val2 is not None:
                val2 = self._get_val_index('value', value_dict)
            if val1 == -1:
                val1 = 0
                logging.info('lost value: %s. candidates: %s', cond[3],
                             ', '.join(value_dict.keys()))
                logging.debug('sql is: %s',
                              json.dumps(sql_json, ensure_ascii=False))
            if val2 == -1:
                val2 = 0
            cond[3] = val1
            cond[4] = val2

        for table_unit in sql_json['from']['table_units']:
            if type(table_unit[1]) is dict:
                self.fix_sql_value(table_unit[1], value_dict)

        for cond in sql_json['where'][::2]:
            _fix_cond_value(cond)
        for cond in sql_json['having'][::2]:
            _fix_cond_value(cond)

        if sql_json['limit'] is not None:
            limit = 'value'
        else:
            limit = 'empty'
        assert limit in value_dict
        sql_json['limit'] = value_dict[limit]

        if sql_json['intersect'] is not None:
            self.fix_sql_value(sql_json['intersect'], value_dict)
        if sql_json['union'] is not None:
            self.fix_sql_value(sql_json['union'], value_dict)
        if sql_json['except'] is not None:
            self.fix_sql_value(sql_json['except'], value_dict)

    def add_item(self, section, sql_json, value_list):
        """add an item"""
        value_dict = {val: idx for idx, val in enumerate(value_list)}
        self.format_sql_value(sql_json, value_dict)

        parsed = self.grammar.parse(sql_json, section)
        self.ast_wrapper.verify_ast(
            parsed)  # will raise AssertionError, if varify failed

        root = parsed
        if section == 'train':
            for token in self._all_tokens(root):
                self.vocab_builder.add_word(token)
            self._record_productions(root)

        item = DecoderSQLItem(tree=root, orig_code=sql_json)
        self.items[section].append(item)
        return item

    def clear_items(self):
        """clear items"""
        self.items = collections.defaultdict(list)

    def _construct_cache_path(self, root_path):
        root_path = Path(root_path)
        self.vocab_path = root_path / 'dec_vocab.json'
        self.observed_productions_path = root_path / 'observed_productions.json'
        self.grammar_rules_path = root_path / 'grammar_rules.json'
        self.grammar_file = root_path / 'grammar.asdl'

    def save(self, save_path):
        """save parsed items to disk"""
        os.makedirs(save_path, exist_ok=True)
        self._construct_cache_path(save_path)

        self.vocab = self.vocab_builder.finish()
        self.vocab.save(self.vocab_path)
        # observed_productions
        self.sum_type_constructors = serialization.to_dict_with_sorted_values(
            self.sum_type_constructors)
        self.field_presence_infos = serialization.to_dict_with_sorted_values(
            self.field_presence_infos, key=str)
        self.seq_lengths = serialization.to_dict_with_sorted_values(
            self.seq_lengths)
        self.primitive_types = sorted(self.primitive_types)
        with open(self.observed_productions_path, 'w') as f:
            json.dump(
                {
                    'sum_type_constructors': self.sum_type_constructors,
                    'field_presence_infos': self.field_presence_infos,
                    'seq_lengths': self.seq_lengths,
                    'primitive_types': self.primitive_types,
                },
                f,
                indent=2,
                sort_keys=True)

        # grammar
        self.all_rules, self.rules_mask = self._calculate_rules()
        with open(self.grammar_rules_path, 'w') as f:
            json.dump(
                {
                    'all_rules': self.all_rules,
                    'rules_mask': self.rules_mask,
                },
                f,
                indent=2,
                sort_keys=True)

        shutil.copy2(self.base_path, self.grammar_file)

    def load(self, grammar_class):
        """load parsed items from disk"""
        self._construct_cache_path(self.base_path)

        self.grammar = grammar_class(self.grammar_file)
        self.ast_wrapper = self.grammar.ast_wrapper
        self.vocab = vocab.Vocab.load(self.vocab_path)

        observed_productions = json.load(open(self.observed_productions_path))
        self.sum_type_constructors = observed_productions[
            'sum_type_constructors']
        self.field_presence_infos = observed_productions['field_presence_infos']
        self.seq_lengths = observed_productions['seq_lengths']
        self.primitive_types = observed_productions['primitive_types']

        grammar = json.load(open(self.grammar_rules_path))
        self.all_rules = serialization.tuplify(grammar['all_rules'])
        self.rules_mask = grammar['rules_mask']

    def _record_productions(self, tree):
        """_record_productions"""
        queue = [(tree, False)]
        while queue:
            node, is_seq_elem = queue.pop()
            node_type = node['_type']

            # Rules of the form:
            # expr -> Attribute | Await | BinOp | BoolOp | ...
            # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
            for type_name in [node_type] + node.get('_extra_types', []):
                if type_name in self.ast_wrapper.constructors:
                    sum_type_name = self.ast_wrapper.constructor_to_sum_type[
                        type_name]
                    if is_seq_elem and self.use_seq_elem_rules:
                        self.sum_type_constructors[sum_type_name +
                                                   '_seq_elem'].add(type_name)
                    else:
                        self.sum_type_constructors[sum_type_name].add(type_name)

            # Rules of the form:
            # FunctionDef
            # -> identifier name, arguments args
            # |  identifier name, arguments args, stmt* body
            # |  identifier name, arguments args, expr* decorator_list
            # |  identifier name, arguments args, expr? returns
            # ...
            # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
            assert node_type in self.ast_wrapper.singular_types
            field_presence_info = get_field_presence_info(
                self.ast_wrapper, node,
                self.ast_wrapper.singular_types[node_type].fields)
            self.field_presence_infos[node_type].add(field_presence_info)

            for field_info in self.ast_wrapper.singular_types[node_type].fields:
                field_value = node.get(field_info.name,
                                       [] if field_info.seq else None)
                to_enqueue = []
                if field_info.seq:
                    # Rules of the form:
                    # stmt* -> stmt
                    #        | stmt stmt
                    #        | stmt stmt stmt
                    self.seq_lengths[field_info.type + '*'].add(
                        len(field_value))
                    to_enqueue = field_value
                else:
                    to_enqueue = [field_value]
                for child in to_enqueue:
                    if isinstance(child,
                                  collections.abc.Mapping) and '_type' in child:
                        queue.append((child, field_info.seq))
                    else:
                        self.primitive_types.add(type(child).__name__)

    def _calculate_rules(self):
        """_calculate_rules"""
        offset = 0

        all_rules = []
        rules_mask = {}

        # Rules of the form:
        # expr -> Attribute | Await | BinOp | BoolOp | ...
        # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
        for parent, children in sorted(self.sum_type_constructors.items()):
            assert not isinstance(children, set)
            rules_mask[parent] = (offset, offset + len(children))
            offset += len(children)
            all_rules += [(parent, child) for child in children]

        # Rules of the form:
        # FunctionDef
        # -> identifier name, arguments args
        # |  identifier name, arguments args, stmt* body
        # |  identifier name, arguments args, expr* decorator_list
        # |  identifier name, arguments args, expr? returns
        # ...
        # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
        for name, field_presence_infos in sorted(
                self.field_presence_infos.items()):
            assert not isinstance(field_presence_infos, set)
            rules_mask[name] = (offset, offset + len(field_presence_infos))
            offset += len(field_presence_infos)
            all_rules += [(name, presence) for presence in field_presence_infos]

        # Rules of the form:
        # stmt* -> stmt
        #        | stmt stmt
        #        | stmt stmt stmt
        for seq_type_name, lengths in sorted(self.seq_lengths.items()):
            assert not isinstance(lengths, set)
            rules_mask[seq_type_name] = (offset, offset + len(lengths))
            offset += len(lengths)
            all_rules += [(seq_type_name, i) for i in lengths]

        return tuple(all_rules), rules_mask

    def _all_tokens(self, root):
        """_all_tokens"""
        queue = [root]
        while queue:
            node = queue.pop()
            type_info = self.ast_wrapper.singular_types[node['_type']]

            for field_info in reversed(type_info.fields):
                field_value = node.get(field_info.name)
                if field_info.type in self.grammar.pointers:
                    pass
                elif field_info.type in self.ast_wrapper.primitive_types:
                    for token in self.grammar.tokenize_field_value(field_value):
                        yield token
                elif isinstance(field_value, (list, tuple)):
                    queue.extend(field_value)
                elif field_value is not None:
                    queue.append(field_value)


if __name__ == "__main__":
    """run some simple test cases"""
    pass
