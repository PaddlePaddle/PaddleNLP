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
"""
Calculating the exact accuracy. For select, where and others schema, it will be
seen as right if has different order. This script refers to https://github.com/taoyds/spider。
"""
import sys
import os
import traceback
import logging
from io import open
import json
import copy
from collections import defaultdict
import re
import six

from text2sql.utils import text_utils

################################
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id)
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, cond_op, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': [(agg_id, val_unit), (agg_id, val_unit), ...]
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [(agg_id, val_unit), ...])
#   'having': condition
#   'limit': None/number(int)
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit',
                   'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

COND_OPS = ('not_in', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

LOGIC_AND_OR = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

CONST_COLUMN = set(['time_now'])

EXPECT_BRACKET_PRE_TOKENS = set(AGG_OPS + SQL_OPS + COND_OPS + CLAUSE_KEYWORDS +
                                ('from', ',', 'distinct'))

g_empty_sql = {
    "select": [],
    "from": {
        "conds": [],
        "table_units": []
    },
    "where": [],
    "groupBy": [],
    "having": [],
    "orderBy": [],
    "limit": None,
    "except": None,
    "intersect": None,
    "union": None
}

g_eval_value = True
g_is_nl2sql_dataset = False


#################################
def tokenize(string):
    """
    Args:

    Returns:
    """
    string = string.replace("\'", "\"").lower()
    assert string.count('"') % 2 == 0, "Unexpected quote"

    def _extract_value(string):
        """extract values in sql"""
        fields = string.split('"')
        for idx, tok in enumerate(fields):
            if idx % 2 == 1:
                fields[idx] = '"%s"' % (tok)
        return fields

    def _resplit(tmp_tokens, fn_split, fn_omit):
        """resplit"""
        new_tokens = []
        for token in tmp_tokens:
            token = token.strip()
            if fn_omit(token):
                new_tokens.append(token)
            elif re.match(r'\d\d\d\d-\d\d(-\d\d)?', token):
                new_tokens.append('"%s"' % (token))
            else:
                new_tokens.extend(fn_split(token))
        return new_tokens

    tokens_tmp = _extract_value(string)

    two_bytes_op = ['==', '!=', '>=', '<=', '<>', '<in>']
    sep1 = re.compile(r'([ \+\-\*/\(\),><;])')  # 单字节运算符
    sep2 = re.compile('(' + '|'.join(two_bytes_op) + ')')  # 多字节运算符
    tokens_tmp = _resplit(tokens_tmp, lambda x: x.split(' '),
                          lambda x: x.startswith('"'))
    tokens_tmp = _resplit(tokens_tmp, lambda x: re.split(sep2, x),
                          lambda x: x.startswith('"'))
    tokens_tmp = _resplit(tokens_tmp, lambda x: re.split(sep1, x),
                          lambda x: x in two_bytes_op or x.startswith('"'))
    tokens = list(
        filter(lambda x: x.strip() not in ('', 'distinct', 'DISTINCT'),
               tokens_tmp))

    def _post_merge(tokens):
        """merge:
              * col name with "(", ")"
              * values with +/-
        """
        idx = 1
        while idx < len(tokens):
            if tokens[idx] == '(' and tokens[
                    idx - 1] not in EXPECT_BRACKET_PRE_TOKENS:
                while idx < len(tokens):
                    tmp_tok = tokens.pop(idx)
                    tokens[idx - 1] += tmp_tok
                    if tmp_tok == ')':
                        break
            elif tokens[idx] in ('+', '-') and tokens[
                    idx - 1] in COND_OPS and idx + 1 < len(tokens):
                tokens[idx] += tokens[idx + 1]
                tokens.pop(idx + 1)
                idx += 1
            else:
                idx += 1
        return tokens

    tokens = _post_merge(tokens)
    return tokens


def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        alias[toks[idx + 1]] = toks[idx - 1]
    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(
            key)
        tables[key] = key
    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    tok = toks[start_idx]
    if tok == "*":
        return start_idx + 1, schema.id_map[tok]
    if tok in CONST_COLUMN:
        return start_idx + 1, tok

    if g_is_nl2sql_dataset:
        fn_check = lambda tok: '.' in tok and tok.startswith('table_')
    else:
        fn_check = lambda tok: '.' in tok
    if fn_check(tok):  # if token is a composite
        alias, col = tok.split('.', 1)
        key = tables_with_alias[alias] + "." + col
        return start_idx + 1, schema.id_map[key]

    assert default_tables is not None and len(
        default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx + 1, schema.id_map[key]

    raise RuntimeError("Error col: {} from {}".format(tok, toks))


def parse_col_unit(toks,
                   start_idx,
                   tables_with_alias,
                   schema,
                   default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema,
                                default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id)

    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema,
                            default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id)


def parse_val_unit(toks,
                   start_idx,
                   tables_with_alias,
                   schema,
                   default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema,
                                    default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema,
                                        default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    if unit_op in (UNIT_OPS.index('+'), UNIT_OPS.index('*')):
        col_unit1, col_unit2 = sorted([col_unit1, col_unit2])

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx + 1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.id_map[key], key


def parse_value(toks,
                start_idx,
                tables_with_alias,
                schema,
                default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    def _force_float(str_num):
        """force float, just for debug"""
        last = ''
        while len(str_num) > 0:
            try:
                n = float(str_num)
                if last == '%':
                    n /= 100
                return n
            except:
                last = str_num[-1]
                str_num = str_num[:-1]
        raise ValueError('not a float number')

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif toks[idx].startswith('"') and toks[idx].endswith(
            '"'):  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val_str = toks[idx]
            #val = float(val_str) if val_str[-1] != '%' else float(val_str[:-1]) / 100
            val = _force_float(val_str)
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')' \
                and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS \
                and toks[end_idx] not in JOIN_KEYWORDS:
                end_idx += 1

            idx, val = parse_col_unit(toks[start_idx:end_idx], 0,
                                      tables_with_alias, schema, default_tables)
            idx = end_idx

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks,
                    start_idx,
                    tables_with_alias,
                    schema,
                    default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []

    while idx < len_:
        agg_id = 0
        if idx < len_ and toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1

        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema,
                                       default_tables)

        op_str = toks[idx]
        if op_str == 'not':
            assert toks[idx + 1] == 'in', '"not" must followed by "in"'
            op_str = 'not_in'
            idx += 1
        assert idx < len_ and op_str in COND_OPS, "Error condition: idx: {}, tok: {}".format(
            idx, op_str)
        op_id = COND_OPS.index(op_str)
        idx += 1
        val1 = val2 = None
        if op_id == COND_OPS.index('between'):
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema,
                                    default_tables)
            assert toks[idx].lower() == 'and'
            idx += 1
            idx, val2 = parse_value(toks, idx, tables_with_alias, schema,
                                    default_tables)
        else:
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema,
                                    default_tables)
            val2 = None

        conds.append((agg_id, op_id, val_unit, val1, val2))

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx]
                           in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

        if idx < len_ and toks[idx] in LOGIC_AND_OR:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(toks,
                 start_idx,
                 tables_with_alias,
                 schema,
                 default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema,
                                       default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, val_units


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []
    last_table = None

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
            last_table = sql['from']['table_units'][0][1].strip('_')
        else:
            if idx < len_ and toks[idx] == 'join':
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(
                toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'], table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias,
                                              schema, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        if idx < len_ and toks[idx] == 'a':
            assert last_table is not None, 'last_table should be a table name strin, not None'
            tables_with_alias['a'] = last_table
            idx += 2
        elif idx < len_ and toks[idx] == 'b':
            assert last_table is not None, 'last_table should be a table name strin, not None'
            tables_with_alias['b'] = last_table
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS
                           or toks[idx] in (")", ";")):
            break

    return [idx, table_units, conds, default_tables]


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema,
                                 default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS
                              or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema,
                                       default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc'  # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS
                              or toks[idx] in (")", ";")):
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema,
                                       default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema,
                                 default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        limit_num = int(toks[idx - 1])
        return idx, limit_num

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema):
    isBlock = False  # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(
        toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(toks, idx, tables_with_alias, schema,
                                       default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema,
                                   default_tables)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema,
                                          default_tables)
    sql['groupBy'] = group_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema,
                                     default_tables)
    sql['having'] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema,
                                          default_tables)
    sql['orderBy'] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(schema, query):
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)

    return sql


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx


#################################

g_db_schema_file = None
g_foreign_key_maps = None


class Evaluator(object):
    """A simple evaluator"""

    def __init__(self, db_schema_file, foreign_key_maps, eval_value=True):
        """init"""
        self.schemas = {}
        self.foreign_key_maps = foreign_key_maps
        self.partial_scores = None
        self.scores = {'all': {'count': 0, 'exact': 0}}
        global g_db_schema_file
        global g_foreign_key_maps
        g_db_schema_file = db_schema_file
        g_foreign_key_maps = foreign_key_maps

        with open(db_schema_file) as ifs:
            databases = json.load(ifs)
            for db in databases:
                self.schemas[db['db_id']] = Schema(db)
        is_nl2sql = all([len(x.schema) == 1 for x in self.schemas.values()])
        if is_nl2sql:
            global g_is_nl2sql_dataset
            g_is_nl2sql_dataset = True

        # number of failed to parse predicted sql query
        self.eval_err_num = 0

        global g_eval_value
        g_eval_value = eval_value
        self.eval_value = eval_value

    def _eval_exact_match(self, pred, gold):
        """eval_exact_match"""
        partial_scores = self.eval_partial_match(pred, gold)
        self.partial_scores = partial_scores

        for _, score in partial_scores.items():
            if score['f1'] != 1:
                return 0

        gold_table_units = gold['from']['table_units']
        pred_table_units = pred['from']['table_units']
        if len(pred_table_units) != len(gold_table_units) or \
                any(map(lambda x: type(x[0][1]) != type(x[1][1]), zip(pred_table_units, gold_table_units))):
            return 0
        if type(gold_table_units[0][1]) is not dict:
            return 1 if sorted(gold_table_units) == sorted(
                pred_table_units) else 0

        # TODO: 严格考虑顺序
        def __eval_from_sql(pred_tables, gold_tables):
            """eval from sql"""
            for pred_table_unit, gold_table_unit in zip(pred_tables,
                                                        gold_tables):
                pred_table_sql = pred_table_unit[1]
                gold_table_sql = gold_table_unit[1]
                _, _, correct = eval_nested(pred_table_sql, gold_table_sql)
                if correct == 0:
                    return 0
            return 1

        correct = __eval_from_sql(pred_table_units, gold_table_units)
        if len(gold_table_units) > 1 and correct == 0:
            return __eval_from_sql(pred_table_units,
                                   list(reversed(gold_table_units)))
        else:
            return correct

        #if len(gold['from']['table_units']) > 0:
        #    gold_tables = sorted(gold['from']['table_units'], key=lambda x: str(x))
        #    pred_tables = sorted(pred['from']['table_units'], key=lambda x: str(x))
        #    return gold_tables == pred_tables
        #return 1

    def eval_exact_match(self, pred, gold):
        """wrapper of evaluate examct match, to process
        `SQL1 intersect/union SQL2` vs `SQL2 intersect/union SQL1`
        """
        score = self._eval_exact_match(pred, gold)
        if score == 1:
            return score

        if gold['union'] is not None:
            new_gold = gold['union']
            gold['union'] = None
            new_gold['union'] = gold
            return self._eval_exact_match(pred, new_gold)
        elif gold['intersect'] is not None:
            new_gold = gold['intersect']
            gold['intersect'] = None
            new_gold['intersect'] = gold
            return self._eval_exact_match(pred, new_gold)
        else:
            return 0

    def eval_partial_match(self, pred, gold):
        """eval partial match"""
        res = {}

        gold_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['select'] = {
            'acc': acc,
            'rec': rec,
            'f1': f1,
            'gold_total': gold_total,
            'pred_total': pred_total
        }
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, gold_total)
        res['select(no AGG)'] = {
            'acc': acc,
            'rec': rec,
            'f1': f1,
            'gold_total': gold_total,
            'pred_total': pred_total
        }

        gold_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['where'] = {
            'acc': acc,
            'rec': rec,
            'f1': f1,
            'gold_total': gold_total,
            'pred_total': pred_total
        }
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, gold_total)
        res['where(no OP)'] = {
            'acc': acc,
            'rec': rec,
            'f1': f1,
            'gold_total': gold_total,
            'pred_total': pred_total
        }

        gold_total, pred_total, cnt = eval_group(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['group(no Having)'] = {
            'acc': acc,
            'rec': rec,
            'f1': f1,
            'gold_total': gold_total,
            'pred_total': pred_total
        }

        gold_total, pred_total, cnt = eval_having(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['group'] = {
            'acc': acc,
            'rec': rec,
            'f1': f1,
            'gold_total': gold_total,
            'pred_total': pred_total
        }

        gold_total, pred_total, cnt = eval_order(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['order'] = {
            'acc': acc,
            'rec': rec,
            'f1': f1,
            'gold_total': gold_total,
            'pred_total': pred_total
        }

        gold_total, pred_total, cnt = eval_and_or(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['and/or'] = {
            'acc': acc,
            'rec': rec,
            'f1': f1,
            'gold_total': gold_total,
            'pred_total': pred_total
        }

        gold_total, pred_total, cnt = eval_IUEN(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['IUEN'] = {
            'acc': acc,
            'rec': rec,
            'f1': f1,
            'gold_total': gold_total,
            'pred_total': pred_total
        }

        gold_total, pred_total, cnt = eval_keywords(pred, gold)
        acc, rec, f1 = get_scores(cnt, pred_total, gold_total)
        res['keywords'] = {
            'acc': acc,
            'rec': rec,
            'f1': f1,
            'gold_total': gold_total,
            'pred_total': pred_total
        }

        return res

    def evaluate_one(self, db_id, gold_query, pred_query):
        """evaluate one predicted result, and cache evaluating info

        Args:
            db (TYPE): NULL
            gold_query (TYPE): NULL
            pred_query (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        self.scores['all']['count'] += 1

        schema = self.schemas[db_id]
        kmap = self.foreign_key_maps[db_id]

        gold_query = gold_query.replace('==', '=')
        gold_sql = get_sql(schema, gold_query)
        # rebuild sql for value evaluation
        g_valid_col_units = build_valid_col_units(
            gold_sql['from']['table_units'], schema)
        gold_sql = rebuild_sql_col(g_valid_col_units, gold_sql, kmap,
                                   self.eval_value)

        is_parse_error = False
        try:
            pred_sql = get_sql(schema, pred_query)

            p_valid_col_units = build_valid_col_units(
                pred_sql['from']['table_units'], schema)
            pred_sql = rebuild_sql_col(p_valid_col_units, pred_sql, kmap,
                                       self.eval_value)
        except Exception as e:
            # If pred_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            pred_sql = g_empty_sql
            self.eval_err_num += 1
            is_parse_error = True

        exact_score = self.eval_exact_match(pred_sql, gold_sql)
        if exact_score == 0:
            logging.debug("error instance %s:\npred: %s\ngold: %s" %
                          (db_id, pred_query, gold_query))
        self.scores['all']['exact'] += exact_score
        return {
            'gold': gold_query,
            'pred': pred_query,
            'correct': int(exact_score),
            'parse_error': int(is_parse_error)
        }

    def finalize(self):
        """
        Returns: TODO

        Raises: NULL
        """
        self.scores['all']['exact'] /= self.scores['all']['count']
        return self.scores


class Schema(object):
    """
    Simple schema which maps table&column to a unique identifier
    """

    def __init__(self, db):
        """init"""
        self._schema = self._build_schema(db)
        self._id_map = self._map(self._schema)

    @property
    def schema(self):
        """_schema property"""
        return self._schema

    @property
    def id_map(self):
        """_id_map property"""
        return self._id_map

    def _build_schema(self, db):
        """build <table, list of columns> schema by input db

        Args:
            db (dict): NULL

        Returns: TODO

        Raises: NULL
        """
        tables = [
            x.lower() for x in db.get('table_names_original', db['table_names'])
        ]
        dct_table2cols = defaultdict(list)
        for table_id, column in db.get('column_names_original',
                                       db['column_names']):
            if table_id < 0:
                continue
            dct_table2cols[tables[table_id]].append(column.lower())
        return dct_table2cols

    def _map(self, schema):
        """map"""
        id_map = {'*': "__all__"}
        for key, vals in schema.items():
            for val in vals:
                id_map[
                    key.lower() + "." +
                    val.lower()] = "__" + key.lower() + "." + val.lower() + "__"

        for key in schema:
            id_map[key.lower()] = "__" + key.lower() + "__"

        return id_map


def get_scores(count, pred_total, gold_total):
    """
    Args:

    Returns:
    """
    if pred_total != gold_total:
        return 0, 0, 0
    elif count == pred_total:
        return 1, 1, 1
    return 0, 0, 0


def eval_sel(pred, gold):
    """
    Args:

    Returns:
    """
    pred_sel = copy.deepcopy(pred['select'])
    gold_sel = copy.deepcopy(gold['select'])
    gold_wo_agg = [unit[1] for unit in gold_sel]
    pred_total = len(pred_sel)
    gold_total = len(gold_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in gold_sel:
            cnt += 1
            gold_sel.remove(unit)
        if unit[1] in gold_wo_agg:
            cnt_wo_agg += 1
            gold_wo_agg.remove(unit[1])

    return [gold_total, pred_total, cnt, cnt_wo_agg]


def eval_nested_cond(pred_cond, gold_cond):
    """

    Args:
        pred_cond (TYPE): NULL
        gold_cond (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    if pred_cond[:3] != gold_cond[:3] or type(pred_cond[3]) is not dict:
        return 0

    _, _, correct = eval_nested(pred_cond[3], gold_cond[3])
    if correct == 0:
        return 0

    return pred_cond[4] == gold_cond[4]


def eval_cond(pred, gold):

    def _equal(p, g):
        if str(p) == str(g):
            return True
        p = p.strip('"\'') if type(p) is str else p
        g = g.strip('"\'') if type(g) is str else g
        if text_utils.is_float(p) and text_utils.is_float(g) and float(
                p) == float(g):
            return True
        return False

    if type(gold[3]) is dict:
        return eval_nested_cond(pred, gold)

    if pred[:3] != gold[:3]:
        return 0

    if _equal(pred[3], gold[3]) and _equal(pred[4], gold[4]):
        return 1
    else:
        return 0


def eval_where(pred, gold):
    pred_conds = list(
        sorted([unit for unit in pred['where'][::2]],
               key=lambda x: [str(i) for i in x]))
    gold_conds = list(
        sorted([unit for unit in gold['where'][::2]],
               key=lambda x: [str(i) for i in x]))
    #gold_wo_agg = [unit[2] for unit in gold_conds]
    pred_total = len(pred_conds)
    gold_total = len(gold_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit_p, unit_g in zip(pred_conds, gold_conds):
        cnt += eval_cond(unit_p, unit_g)

        if unit_p[2] == unit_g[2]:
            cnt_wo_agg += 1

    #for unit in pred_conds:
    #    if unit in gold_conds:
    #        cnt += 1
    #        gold_conds.remove(unit)
    #    if unit[2] in gold_wo_agg:
    #        cnt_wo_agg += 1
    #        gold_wo_agg.remove(unit[2])
    return [gold_total, pred_total, cnt, cnt_wo_agg]
    #return [gold_total, pred_total, cnt, gold_total]


def eval_group(pred, gold):
    pred_cols = [unit[1] for unit in pred['groupBy']]
    gold_cols = [unit[1] for unit in gold['groupBy']]
    pred_total = len(pred_cols)
    gold_total = len(gold_cols)
    cnt = 0
    pred_cols = [
        pred.split(".")[1] if "." in pred else pred for pred in pred_cols
    ]
    gold_cols = [
        gold.split(".")[1] if "." in gold else gold for gold in gold_cols
    ]
    for col in pred_cols:
        if col in gold_cols:
            cnt += 1
            gold_cols.remove(col)
    return [gold_total, pred_total, cnt]


def eval_having(pred, gold):
    """and/or will be evaluate in other branch
    """
    if len(pred['having']) != len(gold['having']):
        return [1, 1, 0]

    pred_total = len(pred['having'][::2])
    gold_total = len(gold['having'][::2])
    cnt = 0
    for pred_cond, gold_cond in zip(sorted(pred['having'][::2]),
                                    sorted(gold['having'][::2])):
        if eval_cond(pred_cond, gold_cond) == 1:
            cnt += 1

    return [gold_total, pred_total, cnt]


def eval_order(pred, gold):
    pred_total = gold_total = cnt = 0
    if len(pred['orderBy']) > 0:
        pred_total = 1
    if len(gold['orderBy']) > 0:
        gold_total = 1

    if len(gold['orderBy']) > 0 and pred['orderBy'] == gold['orderBy'] and pred[
            'limit'] == gold['limit']:
        cnt = 1

    return [gold_total, pred_total, cnt]


def eval_and_or(pred, gold):

    def _extract(conds):
        """extract condition and/or"""
        op_set = set()
        for i in range(1, len(conds) - 1, 2):
            left = conds[i - 1][:3]
            right = conds[i + 1][:3]
            left, right = list(sorted([left, right]))
            op_set.add(f'{left}{conds[i].lower()}{right}')
        return op_set

    # eval where and/or
    pred_op_set = _extract(pred['where'])
    gold_op_set = _extract(gold['where'])
    if pred_op_set != gold_op_set:
        return [1, 1, 0]

    # eval having and/or
    pred_op_set = _extract(pred['having'])
    gold_op_set = _extract(gold['having'])
    if pred_op_set != gold_op_set:
        return [1, 1, 0]

    return [1, 1, 1]


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql[
            'having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    ##
    for from_nest_sql in [
            table_unit[1] for table_unit in sql['from']['table_units']
            if table_unit[0] == 'sql'
    ]:
        nested.append(from_nest_sql)

    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def eval_nested(pred, gold):
    gold_total = 0
    pred_total = 0
    cnt = 0
    if pred is not None:
        pred_total += 1
    if gold is not None:
        gold_total += 1
    if pred is not None and gold is not None:
        cnt += Evaluator(g_db_schema_file, g_foreign_key_maps,
                         g_eval_value).eval_exact_match(pred, gold)
    return [gold_total, pred_total, cnt]


def eval_IUEN(pred, gold):
    lt1, pt1, cnt1 = eval_nested(pred['intersect'], gold['intersect'])
    lt2, pt2, cnt2 = eval_nested(pred['except'], gold['except'])
    lt3, pt3, cnt3 = eval_nested(pred['union'], gold['union'])
    gold_total = lt1 + lt2 + lt3
    pred_total = pt1 + pt2 + pt3
    cnt = cnt1 + cnt2 + cnt3
    return [gold_total, pred_total, cnt]


def get_keywords(sql):
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')

    # or keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    ## TODO
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql[
        'having'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([
            cond_unit
            for cond_unit in cond_units if cond_unit[1] == COND_OPS.index('in')
    ]) > 0:
        res.add('in')

    # like keyword
    if len([
            cond_unit for cond_unit in cond_units
            if cond_unit[1] == COND_OPS.index('like')
    ]) > 0:
        res.add('like')

    return res


def eval_keywords(pred, gold):
    pred_keywords = get_keywords(pred)
    gold_keywords = get_keywords(gold)
    pred_total = len(pred_keywords)
    gold_total = len(gold_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in gold_keywords:
            cnt += 1
    return [gold_total, pred_total, cnt]


# Rebuild SQL functions for foreign key evaluation
def build_valid_col_units(table_units, schema):
    col_ids = [
        table_unit[1] for table_unit in table_units
        if table_unit[0] == TABLE_TYPE['table_unit']
    ]
    prefixs = [col_id[:-2] for col_id in col_ids]
    valid_col_units = []
    for value in schema.id_map.values():
        if '.' in value and value[:value.index('.')] in prefixs:
            valid_col_units.append(value)
    return valid_col_units


def rebuild_col_unit_col(valid_col_units, col_unit, kmap):
    if col_unit is None:
        return col_unit

    agg_id, col_id = col_unit
    if col_id in kmap and col_id in valid_col_units:
        col_id = kmap[col_id]
    return agg_id, col_id


def rebuild_val_unit_col(valid_col_units, val_unit, kmap):
    if val_unit is None:
        return val_unit

    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, kmap)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, kmap)
    return [unit_op, col_unit1, col_unit2]


def rebuild_table_unit_col(valid_col_units, table_unit, kmap, eval_value=True):
    if table_unit is None:
        return table_unit

    table_type, col_unit_or_sql = table_unit
    if isinstance(col_unit_or_sql, dict):
        col_unit_or_sql = rebuild_sql_col(valid_col_units, col_unit_or_sql,
                                          kmap, eval_value)
    elif isinstance(col_unit_or_sql, tuple):  ## useless
        col_unit_or_sql = rebuild_col_unit_col(valid_col_units, col_unit_or_sql,
                                               kmap)
    return table_type, col_unit_or_sql


def rebuild_cond_unit_col(valid_col_units, cond_unit, kmap, eval_value):
    if cond_unit is None:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    if type(val1) is dict:
        rebuild_sql_col(valid_col_units, val1, kmap, eval_value)
    if not eval_value:
        if type(val1) is not dict:
            val1 = '1'
        if type(val2) is not dict and val2 is not None:
            val2 = '2'
    val_unit = rebuild_val_unit_col(valid_col_units, val_unit, kmap)
    return [not_op, op_id, val_unit, val1, val2]


def rebuild_condition_col(valid_col_units, condition, kmap, eval_value):
    for idx in range(len(condition)):
        if idx % 2 == 0:
            condition[idx] = rebuild_cond_unit_col(valid_col_units,
                                                   condition[idx], kmap,
                                                   eval_value)
    return condition


def rebuild_select_col(valid_col_units, sel, kmap):
    if sel is None:
        return sel
    new_list = []
    for it in sel:
        agg_id, val_unit = it
        new_list.append(
            (agg_id, rebuild_val_unit_col(valid_col_units, val_unit, kmap)))
    return new_list


def rebuild_from_col(valid_col_units, from_, kmap, eval_value=True):
    if from_ is None:
        return from_

    fn_proc = lambda x: rebuild_table_unit_col(valid_col_units, x, kmap,
                                               eval_value)
    from_['table_units'] = [
        fn_proc(table_unit) for table_unit in from_['table_units']
    ]
    from_['conds'] = rebuild_condition_col(valid_col_units, from_['conds'],
                                           kmap, True)
    return from_


def rebuild_group_by_col(valid_col_units, group_by, kmap):
    if group_by is None:
        return group_by

    return [
        rebuild_col_unit_col(valid_col_units, col_unit, kmap)
        for col_unit in group_by
    ]


def rebuild_order_by_col(valid_col_units, order_by, kmap):
    if order_by is None or len(order_by) == 0:
        return order_by

    direction, val_units = order_by
    new_val_units = [(agg_id,
                      rebuild_val_unit_col(valid_col_units, val_unit, kmap))
                     for agg_id, val_unit in val_units]
    return direction, new_val_units


def rebuild_sql_col(valid_col_units, sql, kmap, eval_value):
    if sql is None:
        return sql

    sql['select'] = rebuild_select_col(valid_col_units, sql['select'], kmap)
    sql['from'] = rebuild_from_col(valid_col_units, sql['from'], kmap,
                                   eval_value)
    sql['where'] = rebuild_condition_col(valid_col_units, sql['where'], kmap,
                                         eval_value)
    sql['groupBy'] = rebuild_group_by_col(valid_col_units, sql['groupBy'], kmap)
    sql['orderBy'] = rebuild_order_by_col(valid_col_units, sql['orderBy'], kmap)
    sql['having'] = rebuild_condition_col(valid_col_units, sql['having'], kmap,
                                          eval_value)
    sql['intersect'] = rebuild_sql_col(valid_col_units, sql['intersect'], kmap,
                                       eval_value)
    sql['except'] = rebuild_sql_col(valid_col_units, sql['except'], kmap,
                                    eval_value)
    sql['union'] = rebuild_sql_col(valid_col_units, sql['union'], kmap,
                                   eval_value)
    if not eval_value:
        if sql['limit'] is None or int(sql['limit']) <= 0:
            sql['limit'] = 0
        else:
            sql['limit'] = 1

    return sql


def build_foreign_key_map(entry):
    cols_orig = entry["column_names_original"]
    tables_orig = entry["table_names_original"]

    # rebuild cols corresponding to idmap in Schema
    cols = []
    for col_orig in cols_orig:
        if col_orig[0] >= 0:
            t = tables_orig[col_orig[0]]
            c = col_orig[1]
            cols.append("__" + t.lower() + "." + c.lower() + "__")
        else:
            cols.append("__all__")

    def keyset_in_list(k1, k2, k_list):
        """keyset_in_list"""
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set

    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]

    return foreign_key_map


def build_foreign_key_map_from_json(table):
    with open(table) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry['db_id']] = build_foreign_key_map(entry)
    return tables


if __name__ == "__main__":
    pass
