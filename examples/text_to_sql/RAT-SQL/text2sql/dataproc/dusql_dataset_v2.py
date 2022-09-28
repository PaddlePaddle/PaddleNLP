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
import attr
import networkx as nx
import pickle
from pathlib import Path
import tqdm

import numpy as np
import paddle

from text2sql.utils import text_utils
from text2sql.utils import linking_utils

g_ernie_input_parser = None
g_match_score_threshold = 0.3


@attr.s
class DuSQLItem:
    text = attr.ib()
    code = attr.ib()
    schema = attr.ib()
    orig = attr.ib()
    orig_schema = attr.ib()


@attr.s
class Column:
    id = attr.ib()
    table = attr.ib()
    name = attr.ib()
    orig_name = attr.ib()
    dtype = attr.ib()
    cells = attr.ib(factory=list)
    foreign_key_for = attr.ib(default=None)


@attr.s
class Table:
    id = attr.ib()
    name = attr.ib()
    orig_name = attr.ib()
    columns = attr.ib(factory=list)
    primary_keys = attr.ib(factory=list)
    primary_keys_id = attr.ib(factory=list)
    foreign_keys_tables = attr.ib(factory=set)


@attr.s
class DB:
    db_id = attr.ib()
    tables = attr.ib()
    columns = attr.ib()
    foreign_key_graph = attr.ib()
    orig = attr.ib()
    connection = attr.ib(default=None)


def _extract_column_cells(table_names, tables_content):
    lst_column_cells = [table_names]

    for table_name in table_names:
        table_info = tables_content.get(table_name, None)
        if table_info is None:
            return None
        rows = table_info.get('cell', [])
        if len(rows) == 0:
            rows = [[] for _ in tables_content[table_name]['header']]
            lst_column_cells.extend(rows)
        else:
            lst_column_cells.extend(list(zip(*rows)))

    return lst_column_cells


def load_tables(schema_file, content_file):
    """load tables from json files"""
    schemas = {}
    eval_foreign_key_maps = {}

    with open(schema_file) as ifs_schema, open(content_file) as ifs_content:
        lst_schema = json.load(ifs_schema)
        dct_content = {x['db_id']: x for x in json.load(ifs_content)}

        for schema_dict in lst_schema:
            db_id = schema_dict['db_id']

            contents = dct_content[db_id]
            lst_column_cells = _extract_column_cells(schema_dict['table_names'],
                                                     contents['tables'])
            if lst_column_cells is None:
                lst_column_cells = [[] for _ in schema_dict['column_names']]
            assert len(lst_column_cells) == len(schema_dict['column_names'])

            if 'table_names_original' not in schema_dict:
                schema_dict['table_names_original'] = schema_dict['table_names']
            if 'column_names_original' not in schema_dict:
                schema_dict['column_names_original'] = schema_dict[
                    'column_names']
            tables = tuple(
                Table(id=i, name=text_utils.wordseg(name), orig_name=orig_name)
                for i, (name, orig_name) in enumerate(
                    zip(schema_dict['table_names'],
                        schema_dict['table_names_original'])))
            columns = tuple(
                Column(
                    id=i,
                    table=tables[table_id] if table_id >= 0 else None,
                    name=text_utils.wordseg(col_name),
                    orig_name=orig_col_name,
                    dtype=col_type,
                    # 1. drop data with length > 20
                    # 2. ID is startswith item_
                    cells=[
                        x for x in set([str(c) for c in lst_column_cells[i]])
                        if len(x) <= 20 or x.startswith('item_')
                    ],
                ) for i, ((table_id, col_name), (_, orig_col_name),
                          col_type) in enumerate(
                              zip(schema_dict['column_names'],
                                  schema_dict['column_names_original'],
                                  schema_dict['column_types'])))

            # Link columns to tables
            for column in columns:
                if column.table:
                    column.table.columns.append(column)

            # Register primary keys
            for column_id in schema_dict['primary_keys']:
                column = columns[column_id]
                column.table.primary_keys.append(column)
                column.table.primary_keys_id.append(column_id)

            # Register foreign keys
            foreign_key_graph = nx.DiGraph()
            for source_column_id, dest_column_id in schema_dict['foreign_keys']:
                source_column = columns[source_column_id]
                dest_column = columns[dest_column_id]
                source_column.foreign_key_for = dest_column
                columns[source_column_id].table.foreign_keys_tables.add(
                    dest_column_id)
                foreign_key_graph.add_edge(source_column.table.id,
                                           dest_column.table.id,
                                           columns=(source_column_id,
                                                    dest_column_id))
                foreign_key_graph.add_edge(dest_column.table.id,
                                           source_column.table.id,
                                           columns=(dest_column_id,
                                                    source_column_id))

            schemas[db_id] = DB(db_id, tables, columns, foreign_key_graph,
                                schema_dict)
            # TODO
            ##eval_foreign_key_maps[db_id] = evaluation.build_foreign_key_map(schema_dict)

    return schemas, eval_foreign_key_maps


class DuSQLExample(object):
    """Define struct of one DuSQL example, and its processing methods"""

    def __init__(self, json_example, db, input_encoder):
        super(DuSQLExample, self).__init__()

        self.orig = json_example
        self.question = json_example['question']
        self.question_id = json_example['question_id']
        self.columns = db.columns
        self.tables = db.tables
        self.db = db

        self.column_match_cells = self._filter_match_values(
            json_example['match_values'])

        ernie_inputs = input_encoder.encode(self.question, db,
                                            self.column_match_cells)
        self.token_ids = ernie_inputs.token_ids
        self.sent_ids = ernie_inputs.sent_ids
        self.table_indexes = ernie_inputs.table_indexes
        self.column_indexes = ernie_inputs.column_indexes
        self.value_indexes = ernie_inputs.value_indexes
        self.values = ernie_inputs.value_list

        self.token_mapping = ernie_inputs.token_mapping
        self.question_tokens = ernie_inputs.orig_question_tokens
        self.candi_nums = ernie_inputs.candi_nums
        self.relations = self._compute_relations()

    def _filter_match_values(self, match_values_info):
        """filter by match score
        """
        lst_result = []
        for column_values in match_values_info:
            filtered_results = []
            for value, score in column_values:
                if score > g_match_score_threshold:
                    filtered_results.append(value)
                else:  # column_values should ordered by score
                    break
            lst_result.append(filtered_results)
        return lst_result

    def _compute_relations(self):
        schema_linking_results = self._linking_wrapper(
            linking_utils.compute_schema_linking)
        cell_value_linking_results = self._linking_wrapper(
            linking_utils.compute_cell_value_linking)
        link_info_dict = {
            'sc_link': schema_linking_results,
            'cv_link': cell_value_linking_results
        }

        q_len = self.column_indexes[0] - 2
        c_len = len(self.columns)
        t_len = len(self.tables)
        total_len = q_len + c_len + t_len
        relation_matrix = linking_utils.build_relation_matrix(
            link_info_dict, total_len, q_len, c_len, list(range(c_len + 1)),
            list(range(t_len + 1)), self.db)
        return relation_matrix

    def _linking_wrapper(self, fn_linking):
        """wrapper for linking function, do linking and id convert
        """
        link_result = fn_linking(self.question_tokens, self.db)

        # convert words id to BERT word pieces id
        new_result = {}
        for m_name, matches in link_result.items():
            new_match = {}
            for pos_str, match_type in matches.items():
                qid_str, col_tab_id_str = pos_str.split(',')
                qid, col_tab_id = int(qid_str), int(col_tab_id_str)
                for real_qid in self.token_mapping[qid]:
                    new_match[f'{real_qid},{col_tab_id}'] = match_type
            new_result[m_name] = new_match
        return new_result

    def __repr__(self):
        """format for reviewing
        """
        return str(self.__dict__)


class DuSQLDatasetV2(paddle.io.Dataset):
    """implement of DuSQL dataset for training/evaluating"""

    def __init__(self,
                 name,
                 db_file,
                 data_file,
                 input_encoder,
                 label_encoder,
                 is_cached=False,
                 schema_file=None,
                 has_label=True):
        super(DuSQLDatasetV2, self).__init__()

        self.name = name
        self.input_encoder = input_encoder
        self.label_encoder = label_encoder
        self.db_schema_file = schema_file
        self.has_label = has_label
        self._qid2index = {}

        if is_cached:
            self.db_dict, self._examples = None, None
            self.load(db_file, data_file)
        else:
            schema_file, content_file = db_file
            self.db_dict, _ = load_tables(schema_file, content_file)
            self._examples = []
            match_value_file = Path(os.path.dirname(data_file)) / (
                'match_values_' + os.path.basename(data_file))
            if not match_value_file.exists():
                raise FileNotFoundError('match value file not found: ' +
                                        str(match_value_file))
            with open(data_file) as ifs_data, open(
                    match_value_file) as ifs_mval:
                self.collate_examples(json.load(ifs_data), json.load(ifs_mval))

    def collate_examples(self, orig_examples, match_values):
        """collate examples, and append to self._examples
        """
        for idx, (item, m_val) in tqdm.tqdm(
                enumerate(zip(orig_examples, match_values))):
            if 'question_id' in item:
                assert item['question_id'] == m_val['question_id'], \
                        f'data no match: {item["question_id"]} != {m_val["question_id"]}'
            item['match_values'] = m_val['match_values']
            db = self.db_dict[item['db_id']]
            if not self.input_encoder.check(item, db):
                logging.warning(
                    f'check failed: db_id={item["db_id"]}, question={item["question"]}'
                )
                continue
            if 'question_id' not in item:
                item['question_id'] = f'qid{idx:06d}'
            inputs = DuSQLExample(item, db, self.input_encoder)
            if 'sql' not in item or type(
                    item['sql']) is not dict or not self.has_label:
                outputs = None
            else:
                outputs = self.label_encoder.add_item(self.name, item['sql'],
                                                      inputs.values)
            self._qid2index[item['question_id']] = len(self._examples)
            self._examples.append([inputs, outputs])

    def save(self, save_dir, save_db=True):
        """save data to disk

        Args:
            save_dir (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        os.makedirs(save_dir, exist_ok=True)
        if save_db:
            with open(Path(save_dir) / 'db.pkl', 'wb') as ofs:
                pickle.dump(self.db_dict, ofs)
        with open(Path(save_dir) / f'{self.name}.pkl', 'wb') as ofs:
            pickle.dump([self._examples, self._qid2index], ofs)

    def load(self, db_file, data_file):
        """load data from disk
        """
        with open(db_file, 'rb') as ifs:
            self.db_dict = pickle.load(ifs)
        with open(data_file, 'rb') as ifs:
            self._examples, self._qid2index = pickle.load(ifs)

    def get_by_qid(self, qid):
        """
        """
        index = self._qid2index[qid]
        return self._examples[index]

    def __getitem__(self, idx):
        """get one example
        """
        return self._examples[idx]

    def __len__(self):
        """size of data examples
        """
        return len(self._examples)


if __name__ == "__main__":
    """run simple tests"""
    if len(sys.argv) != 5:
        print("usage: %s schema content data grammar_file" % (sys.argv[0]))
        sys.exit(1)
