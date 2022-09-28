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
import re
import itertools

import numpy as np

from text2sql.utils import text_utils

# the max matching ngram
g_linking_ngrams_n = 5

STOPWORDS = set([
    "的", "是", "，", "？", "有", "多少", "哪些", "我", "什么", "你", "知道", "啊", "一下", "吗",
    "在", "请问", "或者", "想", "和", "为", "帮", "那个", "你好", "这", "了", "并且", "都", "呢",
    "呀", "哪个", "还有", "这个", "-", "项目", "我查", "就是", "它", "要求", "谁", "了解", "告诉",
    "时候", "个", "能", "那", "人", "问", "中", "可以", "一共", "哪", "麻烦", "叫", "想要", "《",
    "》", "分别"
])


def clamp(value, abs_max):
    """clamp value"""
    value = max(-abs_max, value)
    value = min(abs_max, value)
    return value


class Relations(object):
    """Docstring for Relations. """

    def __init__(self,
                 qq_max_dist=2,
                 cc_foreign_key=True,
                 cc_table_match=True,
                 cc_max_dist=2,
                 ct_foreign_key=True,
                 ct_table_match=True,
                 tc_table_match=True,
                 tc_foreign_key=True,
                 tt_max_dist=2,
                 tt_foreign_key=True,
                 merge_types=False,
                 sc_link=True,
                 cv_link=True):
        super(Relations, self).__init__()

        self.qq_max_dist = qq_max_dist
        self.cc_foreign_key = cc_foreign_key
        self.cc_table_match = cc_table_match
        self.cc_max_dist = cc_max_dist
        self.ct_foreign_key = ct_foreign_key
        self.ct_table_match = ct_table_match
        self.tc_table_match = tc_table_match
        self.tc_foreign_key = tc_foreign_key
        self.tt_max_dist = tt_max_dist
        self.tt_foreign_key = tt_foreign_key
        self.merge_types = merge_types
        self.sc_link = sc_link
        self.cv_link = cv_link

        self.relation_ids = {}

        def add_relation(name):
            self.relation_ids[name] = len(self.relation_ids)
            logging.debug('relation: %s --> %d', name, self.relation_ids[name])

        ##< TODO: add_relation('[UNK]')

        def add_rel_dist(name, max_dist):
            for i in range(-max_dist, max_dist + 1):
                add_relation((name, i))

        add_rel_dist('qq_dist', qq_max_dist)

        add_relation('qc_default')
        # if qc_token_match:
        #    add_relation('qc_token_match')

        add_relation('qt_default')
        # if qt_token_match:
        #    add_relation('qt_token_match')

        add_relation('cq_default')
        # if cq_token_match:
        #    add_relation('cq_token_match')

        add_relation('cc_default')
        if cc_foreign_key:
            add_relation('cc_foreign_key_forward')
            add_relation('cc_foreign_key_backward')
        if cc_table_match:
            add_relation('cc_table_match')
        add_rel_dist('cc_dist', cc_max_dist)

        add_relation('ct_default')
        if ct_foreign_key:
            add_relation('ct_foreign_key')
        if ct_table_match:
            add_relation('ct_primary_key')
            add_relation('ct_table_match')
            add_relation('ct_any_table')

        add_relation('tq_default')
        # if cq_token_match:
        #    add_relation('tq_token_match')

        add_relation('tc_default')
        if tc_table_match:
            add_relation('tc_primary_key')
            add_relation('tc_table_match')
            add_relation('tc_any_table')
        if tc_foreign_key:
            add_relation('tc_foreign_key')

        add_relation('tt_default')
        if tt_foreign_key:
            add_relation('tt_foreign_key_forward')
            add_relation('tt_foreign_key_backward')
            add_relation('tt_foreign_key_both')
        add_rel_dist('tt_dist', tt_max_dist)

        # schema linking relations
        # forward_backward
        if sc_link:
            add_relation('qcCEM')
            add_relation('cqCEM')
            add_relation('qtTEM')
            add_relation('tqTEM')
            add_relation('qcCPM')
            add_relation('cqCPM')
            add_relation('qtTPM')
            add_relation('tqTPM')

        if cv_link:
            add_relation("qcNUMBER")
            add_relation("cqNUMBER")
            add_relation("qcTIME")
            add_relation("cqTIME")
            add_relation("qcCELLMATCH")
            add_relation("cqCELLMATCH")

        if merge_types:
            assert not cc_foreign_key
            assert not cc_table_match
            assert not ct_foreign_key
            assert not ct_table_match
            assert not tc_foreign_key
            assert not tc_table_match
            assert not tt_foreign_key

            assert cc_max_dist == qq_max_dist
            assert tt_max_dist == qq_max_dist

            add_relation('xx_default')
            self.relation_ids['qc_default'] = self.relation_ids['xx_default']
            self.relation_ids['qt_default'] = self.relation_ids['xx_default']
            self.relation_ids['cq_default'] = self.relation_ids['xx_default']
            self.relation_ids['cc_default'] = self.relation_ids['xx_default']
            self.relation_ids['ct_default'] = self.relation_ids['xx_default']
            self.relation_ids['tq_default'] = self.relation_ids['xx_default']
            self.relation_ids['tc_default'] = self.relation_ids['xx_default']
            self.relation_ids['tt_default'] = self.relation_ids['xx_default']

            if sc_link:
                self.relation_ids['qcCEM'] = self.relation_ids['xx_default']
                self.relation_ids['qcCPM'] = self.relation_ids['xx_default']
                self.relation_ids['qtTEM'] = self.relation_ids['xx_default']
                self.relation_ids['qtTPM'] = self.relation_ids['xx_default']
                self.relation_ids['cqCEM'] = self.relation_ids['xx_default']
                self.relation_ids['cqCPM'] = self.relation_ids['xx_default']
                self.relation_ids['tqTEM'] = self.relation_ids['xx_default']
                self.relation_ids['tqTPM'] = self.relation_ids['xx_default']
            if cv_link:
                self.relation_ids["qcNUMBER"] = self.relation_ids['xx_default']
                self.relation_ids["cqNUMBER"] = self.relation_ids['xx_default']
                self.relation_ids["qcTIME"] = self.relation_ids['xx_default']
                self.relation_ids["cqTIME"] = self.relation_ids['xx_default']
                self.relation_ids["qcCELLMATCH"] = self.relation_ids[
                    'xx_default']
                self.relation_ids["cqCELLMATCH"] = self.relation_ids[
                    'xx_default']

            for i in range(-qq_max_dist, qq_max_dist + 1):
                self.relation_ids['cc_dist', i] = self.relation_ids['qq_dist',
                                                                    i]
                self.relation_ids['tt_dist', i] = self.relation_ids['tt_dist',
                                                                    i]

        logging.info("relations num is: %d", len(self.relation_ids))

    def __len__(self):
        """size of relations
        Returns: int
        """
        return len(self.relation_ids)


RELATIONS = Relations()


# schema linking, similar to IRNet
def compute_schema_linking(tokens, db):
    """schema linking
    """

    def partial_match(x_list, y_list):
        """check partial match"""
        x_str = "".join(x_list)
        y_str = "".join(y_list)
        if x_str in STOPWORDS:
            return False
        if re.match("%s" % re.escape(x_str), y_str):
            assert x_str in y_str
            return True
        else:
            return False

    def exact_match(x_list, y_list):
        """check exact match"""
        x, y = x_list, y_list
        if type(x) is list:
            x = "".join(x)
        if type(y) is list:
            y = "".join(y)
        return x == y

    def set_q_relation(q_match_dict,
                       q_start,
                       q_match_len,
                       other_id,
                       relation_tag,
                       force=True):
        """set match relation for question
        """
        for q_id in range(q_start, q_start + q_match_len):
            key = f"{q_id},{other_id}"
            if not force and key in q_match_dict:
                continue
            q_match_dict[key] = relation_tag

    columns = [x.name for x in db.columns]
    tables = [x.name for x in db.tables]

    q_col_match = dict()
    q_tab_match = dict()

    col_id2list = dict()
    for col_id, col_item in enumerate(columns):
        col_id2list[col_id] = col_item

    tab_id2list = dict()
    for tab_id, tab_item in enumerate(tables):
        tab_id2list[tab_id] = tab_item

    # 5-gram
    n = g_linking_ngrams_n
    while n > 0:
        for i, n_gram_list in enumerate(text_utils.ngrams(tokens, n)):
            if len("".join(n_gram_list).strip()) == 0:
                continue
            # exact match case
            for col_id, col in col_id2list.items():
                if exact_match(n_gram_list, col):
                    set_q_relation(q_col_match, i, n, col_id, "CEM")
            for tab_id, tab in tab_id2list.items():
                if exact_match(n_gram_list, tab):
                    set_q_relation(q_tab_match, i, n, tab_id, "TEM")

            # partial match case
            for col_id, col in col_id2list.items():
                if partial_match(n_gram_list, col):
                    set_q_relation(q_col_match,
                                   i,
                                   n,
                                   col_id,
                                   "CPM",
                                   force=False)
            for tab_id, tab in tab_id2list.items():
                if partial_match(n_gram_list, tab):
                    set_q_relation(q_tab_match,
                                   i,
                                   n,
                                   tab_id,
                                   "TEM",
                                   force=False)
        n -= 1
    return {"q_col_match": q_col_match, "q_tab_match": q_tab_match}


def compute_cell_value_linking(tokens, db):
    """cell-value linking
    """

    def isnumber(word):
        """check if input is a number"""
        try:
            float(word)
            return True
        except:
            return False

    def check_cell_match(word, cells):
        """check if word partial/exact match one of values
        """
        for cell in cells:
            if word in cell:
                return True
        return False

    num_date_match = {}
    cell_match = {}

    for q_id, word in enumerate(tokens):
        if len(word.strip()) == 0:
            continue
        if word in STOPWORDS:
            continue

        num_flag = isnumber(word)
        for col_id, column in enumerate(db.columns):
            # word is number
            if num_flag:
                if column.dtype in ("number", "real",
                                    "time"):  # TODO fine-grained date
                    rel = 'NUMBER' if column.dtype == 'real' else column.dtype.upper(
                    )
                    num_date_match[f"{q_id},{col_id}"] = rel
            elif column.dtype.lower(
            ) == 'binary':  # binary condition should use special process
                continue
            elif check_cell_match(word, column.cells):
                cell_match[f"{q_id},{col_id}"] = "CELLMATCH"

    cv_link = {"num_date_match": num_date_match, "cell_match": cell_match}
    return cv_link


def _table_id(db, col):
    if col == 0:
        return None
    else:
        return db.columns[col].table.id


def _foreign_key_id(db, col):
    foreign_col = db.columns[col].foreign_key_for
    if foreign_col is None:
        return None
    return foreign_col.id


def _match_foreign_key(db, col, table):
    foreign_key_id = _foreign_key_id(db, col)
    if foreign_key_id is None:
        return None
    return table == _table_id(db, foreign_key_id)


def build_relation_matrix(other_links, total_length, q_length, c_length,
                          c_boundaries, t_boundaries, db):
    """build relation matrix
    """
    sc_link = other_links.get('sc_link', {'q_col_match': {}, 'q_tab_match': {}})
    cv_link = other_links.get('cv_link', {
        'num_date_match': {},
        'cell_match': {}
    })

    # Catalogue which things are where
    loc_types = {}
    for i in range(q_length):
        loc_types[i] = ('question', )

    c_base = q_length
    for c_id, (c_start, c_end) in enumerate(zip(c_boundaries,
                                                c_boundaries[1:])):
        for i in range(c_start + c_base, c_end + c_base):
            loc_types[i] = ('column', c_id)
    t_base = q_length + c_length
    for t_id, (t_start, t_end) in enumerate(zip(t_boundaries,
                                                t_boundaries[1:])):
        for i in range(t_start + t_base, t_end + t_base):
            loc_types[i] = ('table', t_id)

    relations = np.zeros((total_length, total_length), dtype=np.int64)
    for i, j in itertools.product(range(total_length), repeat=2):

        def _set_relation(name):
            """set relation for position (i, j)"""
            relations[i, j] = RELATIONS.relation_ids[name]

        def _get_qc_links(q_id, c_id):
            """get link relation of q and col"""
            coord = "%d,%d" % (q_id, c_id)
            if coord in sc_link["q_col_match"]:
                return sc_link["q_col_match"][coord]
            elif coord in cv_link["cell_match"]:
                return cv_link["cell_match"][coord]
            elif coord in cv_link["num_date_match"]:
                return cv_link["num_date_match"][coord]
            return '_default'

        def _get_qt_links(q_id, c_id):
            """get link relation of q and tab"""
            coord = "%d,%d" % (q_id, c_id)
            if coord in sc_link["q_tab_match"]:
                return sc_link["q_tab_match"][coord]
            else:
                return '_default'

        try:
            i_type, j_type = loc_types[i], loc_types[j]
        except Exception as e:
            logging.error(f'loc_types: {loc_types}. c_boundaries: {c_boundaries}.' + \
                          f'i, j, total_length and q_length: {i}, {j}, {total_length}, {q_length}')
            raise e

        if i_type[0] == 'question':
            ################ relation of question-to-* ####################
            if j_type[0] == 'question':  # relation qq
                _set_relation(('qq_dist', clamp(j - i, RELATIONS.qq_max_dist)))
            elif j_type[0] == 'column':  # relation qc
                j_real = j_type[1]
                rel = _get_qc_links(i, j_real)
                _set_relation('qc' + rel)
            elif j_type[0] == 'table':  # relation qt
                j_real = j_type[1]
                rel = _get_qt_links(i, j_real)
                _set_relation('qt' + rel)
        elif i_type[0] == 'column':
            ################ relation of column-to-* ####################
            if j_type[0] == 'question':  ## relation cq
                i_real = i_type[1]
                rel = _get_qc_links(j, i_real)
                _set_relation('cq' + rel)
            elif j_type[0] == 'column':  ## relation cc
                col1, col2 = i_type[1], j_type[1]
                if col1 == col2:
                    _set_relation(('cc_dist', clamp(j - i,
                                                    RELATIONS.cc_max_dist)))
                else:
                    _set_relation('cc_default')
                    # TODO: foreign keys and table match
                    if RELATIONS.cc_foreign_key:
                        if _foreign_key_id(db, col1) == col2:
                            _set_relation('cc_foreign_key_forward')
                        if _foreign_key_id(db, col2) == col1:
                            _set_relation('cc_foreign_key_backward')
                    if (RELATIONS.cc_table_match
                            and _table_id(db, col1) == _table_id(db, col2)):
                        _set_relation('cc_table_match')
            elif j_type[0] == 'table':  ## relation ct
                col, table = i_type[1], j_type[1]
                _set_relation('ct_default')
                if RELATIONS.ct_foreign_key and _match_foreign_key(
                        db, col, table):
                    _set_relation('ct_foreign_key')
                if RELATIONS.ct_table_match:
                    col_table = _table_id(db, col)
                    if col_table == table:
                        if col in db.columns[col].table.primary_keys_id:
                            _set_relation('ct_primary_key')
                        else:
                            _set_relation('ct_table_match')
                    elif col_table is None:
                        _set_relation('ct_any_table')
        elif i_type[0] == 'table':
            ################ relation of table-to-* ####################
            if j_type[0] == 'question':
                i_real = i_type[1]
                rel = _get_qt_links(j, i_real)
                _set_relation('tq' + rel)
            elif j_type[0] == 'column':
                table, col = i_type[1], j_type[1]
                _set_relation('tc_default')

                if RELATIONS.tc_foreign_key and _match_foreign_key(
                        db, col, table):
                    _set_relation('tc_foreign_key')
                if RELATIONS.tc_table_match:
                    col_table = _table_id(db, col)
                    if col_table == table:
                        if col in db.columns[col].table.primary_keys_id:
                            _set_relation('tc_primary_key')
                        else:
                            _set_relation('tc_table_match')
                    elif col_table is None:
                        _set_relation('tc_any_table')
            elif j_type[0] == 'table':
                table1, table2 = i_type[1], j_type[1]
                if table1 == table2:
                    _set_relation(('tt_dist', clamp(j - i,
                                                    RELATIONS.tt_max_dist)))
                else:
                    _set_relation('tt_default')
                    if RELATIONS.tt_foreign_key:
                        forward = table2 in db.tables[
                            table1].foreign_keys_tables
                        backward = table1 in db.tables[
                            table2].foreign_keys_tables
                        if forward and backward:
                            _set_relation('tt_foreign_key_both')
                        elif forward:
                            _set_relation('tt_foreign_key_forward')
                        elif backward:
                            _set_relation('tt_foreign_key_backward')

    return relations


if __name__ == "__main__":
    """run some simple test cases"""
    q = '帮 我 查 一 下 大众 帕 萨 特 的 轴距 和 能源 类型 分别 是 什么 , 叫 什么 名 ？'.split(' ')
    for i, tok in enumerate(q):
        print(i, tok)
    ##header = Header(['名称', '品牌', '轴距', '能源类型'], ['text', 'text', 'real', 'text'])
    ##print(header.names)
    ##print(compute_schema_linking(q, header))

    ##q = '帮 我 查 一 下 大众 轴距 大于 10 米 的 车 能源 类型 分别 是 什么 ？'.split(' ')
    ##for i, tok in enumerate(q):
    ##    print(i, tok)
    ##rows = [['帕萨特', '大众', '10', '汽油车'],
    ##        ['伊兰特', '现代', '10', '汽油车'],
    ##        ['GL8', '别克', '10', '汽油车']]
    ##table = Table('tid1', 'tname', 'title', header, rows)
    ##print(compute_cell_value_linking(q, table))
