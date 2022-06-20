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
from collections import defaultdict
import re

from paddlenlp.transformers import BertTokenizer

from text2sql.dataproc.dusql_dataset_v2 import load_tables

logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s: %(asctime)s %(filename)s'
                    ' [%(funcName)s:%(lineno)d][%(process)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    filename=None,
                    filemode='a')

g_date_patt = re.compile(
    r'(([0-9]{2})[0-9]{2}年)?[0-9]{1,2}月[0-9]{2}日|([0-9]{2})[0-9]{2}年[0-9]{1,2}月'
)


def get_char_list(sentence):

    def is_ascii(s):
        """check if s is English album or number
        Args:
            s (str): NULL
        Returns: bool
        """
        return ord(s) < 128

    if len(sentence) == 0:
        return []

    lst_result = [sentence[0]]
    last_is_ascii = lst_result[-1].isalnum()
    for char in sentence[1:]:
        if char == ' ':
            last_is_ascii = False
            continue
        elif char == '-':
            last_is_ascii = False
            lst_result.append(char)
            continue

        if is_ascii(char) and last_is_ascii:
            lst_result[-1] += char
            continue

        if is_ascii(char):
            last_is_ascii = True
        else:
            last_is_ascii = False

        lst_result.append(char)

    return tuple(lst_result)


def _format_date_cell(old_cell):
    new_cell = old_cell.rstrip('月日')
    new_cell = new_cell.replace('年', '-')
    new_cell = new_cell.replace('月', '-')
    return new_cell


def _build(cells):
    dct_index = defaultdict(set)
    for cell in set(cells):
        if type(cell) is not str:
            continue
        cell = cell.strip()
        if re.match(g_date_patt, cell):
            cell = _format_date_cell(cell)
        cell_chars = get_char_list(cell.lower())
        dct_index[cell.lower()].add((cell, len(cell_chars)))
        for pos in range(len(cell_chars) - 1):
            bigram = cell_chars[pos:pos + 2]
            ####tri_gram = cell_chars[pos: pos + 3]
            ####four_gram = cell_chars[pos: pos + 4]
            dct_index[bigram].add((cell, len(cell_chars) - 1))
            ####dct_index[tri_gram].add((cell, len(cell_chars) - 2))
            ####dct_index[four_gram].add(cell)
    return dct_index


def build_cell_index(db_dict):
    for db in db_dict.values():
        column_cells = []
        for column in db.columns:
            cell_index = _build(column.cells)
            column_cells.append(cell_index)
        db.column_cells_index = column_cells


def extract_value_from_sql(sql_json, sql_format='dusql'):
    dct_col_values = defaultdict(list)
    if sql_format == 'nl2sql':
        for col, _, val in item['sql']['conds']:
            dct_col_values[col].append(val)
        return dct_col_values

    def _merge_dict(base_dict, extra_dict):
        for k, v in extra_dict.items():
            base_dict[k].extend(v)

    def _extract_value_from_sql_cond(cond, dct_col_values):
        if type(cond[3]) is dict:
            new_col_values = extract_value_from_sql(cond[3])
            _merge_dict(dct_col_values, new_col_values)
            return
        col_id = cond[2][1][1]
        dct_col_values[col_id].append(cond[3])
        if cond[4] is not None:
            dct_col_values[col_id].append(cond[4])

    for table_unit in sql_json['from']['table_units']:
        if type(table_unit[1]) is dict:
            new_col_values = extract_value_from_sql(table_unit[1])
            _merge_dict(dct_col_values, new_col_values)

    for cond in sql_json['where'][::2]:
        _extract_value_from_sql_cond(cond, dct_col_values)
    for cond in sql_json['having'][::2]:
        _extract_value_from_sql_cond(cond, dct_col_values)

    if sql_json['intersect'] is not None:
        new_col_values = extract_value_from_sql(sql_json['intersect'])
        _merge_dict(dct_col_values, new_col_values)
    if sql_json['union'] is not None:
        new_col_values = extract_value_from_sql(sql_json['union'])
        _merge_dict(dct_col_values, new_col_values)
    if sql_json['except'] is not None:
        new_col_values = extract_value_from_sql(sql_json['except'])
        _merge_dict(dct_col_values, new_col_values)

    return dct_col_values


def search_values(query, db, extra_values):
    lst_match_values = []
    for column, cell_index in zip(db.columns, db.column_cells_index):
        if column.id == 0:
            lst_match_values.append([])
            continue
        col_id = column.id

        candi_cnt = defaultdict(float)
        query_chars = get_char_list(query.lower())
        appear_set = set()
        for pos in range(len(query_chars)):
            unigram = query_chars[pos]
            if len(
                    unigram
            ) > 2 and unigram not in appear_set and unigram in cell_index:
                for cell, base in cell_index[unigram]:
                    candi_cnt[cell] += 1.0 / base
            if pos == len(query_chars) - 1:
                break

            bigram = query_chars[pos:pos + 2]
            if bigram not in cell_index:
                continue
            if bigram in appear_set:
                continue
            appear_set.add(bigram)
            for cell, base in cell_index[bigram]:
                candi_cnt[cell] += 1.0 / base

        if extra_values is not None and column.id in extra_values:
            gold_values = extra_values[column.id]
            for gval in gold_values:
                candi_cnt[str(gval)] += 2.0

        lst_match_values.append(
            list(sorted(candi_cnt.items(), key=lambda x: x[1],
                        reverse=True))[:10])

    return lst_match_values


if __name__ == "__main__":
    import argparse
    try:
        arg_parser = argparse.ArgumentParser(
            description="linking candidate values for each column")
        arg_parser.add_argument("input",
                                nargs="?",
                                type=argparse.FileType('r'),
                                default=sys.stdin,
                                help="input file path")
        arg_parser.add_argument("-s",
                                "--db-schema",
                                required=True,
                                help="file path")
        arg_parser.add_argument("-c",
                                "--db-content",
                                required=True,
                                help="file path")
        arg_parser.add_argument("-o",
                                "--output",
                                type=argparse.FileType('w'),
                                default=sys.stdout,
                                help="output file path")
        arg_parser.add_argument('-t',
                                '--is-train',
                                default=False,
                                action="store_true")
        arg_parser.add_argument('-f',
                                '--sql-format',
                                default='dusql',
                                choices=['dusql', 'nl2sql', 'cspider'])
        args = arg_parser.parse_args()

        sys.stderr.write('>>> loading databases...\n')
        dct_db, _ = load_tables(args.db_schema, args.db_content)
        build_cell_index(dct_db)

        sys.stderr.write('>>> extracting values...\n')
        lst_output = []
        for idx, item in enumerate(json.load(args.input)):
            question_id = item.get('question_id', f'qid{idx:06d}')
            question = item['question']
            db_id = item['db_id']
            db = dct_db[db_id]

            extra_values = None
            if args.is_train:
                extra_values = extract_value_from_sql(item['sql'],
                                                      args.sql_format)

            match_values = search_values(question, db, extra_values)
            lst_output.append({
                "question_id": question_id,
                "question": question,
                "db_id": db_id,
                "match_values": match_values
            })

        json.dump(lst_output, args.output, indent=2, ensure_ascii=False)
    except Exception as e:
        traceback.print_exc()
        #logging.critical(traceback.format_exc())
        exit(-1)
