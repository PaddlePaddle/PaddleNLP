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
import numpy as np

g_open_value_predict = False
g_having_agg_threshold = 0.9


class SQL(object):
    """SQL define"""
    op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!=", 4: ">=", 5: "<="}
    agg_sql_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
    conn_sql_dict = {0: "", 1: "and", 2: "or"}
    order_dict = {0: "", 1: "asc", 2: "desc"}
    sel_num_dict = {0: 1, 1: 2, 2: 3, 3: 4}
    #cond_num_dict = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    cond_num_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    group_num_dict = {0: 0, 1: 1}
    group_type_dict = {
        0: 'none',
        1: 'group',
        2: 'group_having',
        3: 'group_order'
    }

    order2id = {"": 0, "asc": 1, "desc": 2}

    num_where_ops = len(op_sql_dict) + 1
    num_agg_ops = len(agg_sql_dict)
    num_cond_ops = len(conn_sql_dict)
    num_order_directions = len(order_dict)
    num_sel_num = len(sel_num_dict)
    num_where_num = len(cond_num_dict)
    num_group_num = len(group_num_dict)
    num_group_type = len(group_type_dict)

    dtype_str = 'text'
    dtype_num = 'real'

    def __init__(self, cond_conn_op: int, agg: list, sel: list, conds: list,
                 **kwargs):
        """doc"""
        self.cond_conn_op = cond_conn_op
        self.sel = []
        self.agg = []
        sel_agg_pairs = sorted(zip(sel, agg), key=lambda x: x[0])
        for col_id, agg_op in sel_agg_pairs:
            self.sel.append(col_id)
            self.agg.append(agg_op)
        self.conds = list(sorted(conds, key=lambda x: x[0]))
        self.order_by = list(sorted(kwargs.get("order_by", [])))
        self.group_by = list(sorted(kwargs.get("group_by", [])))
        self.having = list(sorted(kwargs.get("having", [])))
        order_str = kwargs.get("order_direction", "").lower()
        self.order_direction = self.order2id.get(order_str, 0)
        limit = kwargs.get("limit", None)
        self.limit = '0' if limit is None else str(limit)
        self.sel_num = len(self.sel)
        self.cond_num = len(self.conds)
        self.group_num = len(self.group_by)
        self.group_type = 0
        if len(self.group_by) > 0:
            self.group_type = 1
            if len(self.having) > 0:
                self.group_type = 2
            elif len(self.order_by) > 0 and self.order_by[0][0] > 0:
                self.group_type = 3

    @classmethod
    def from_dict(cls, data: dict):
        """doc"""
        return cls(**data)

    def keys(self):
        """doc"""
        return [
            'cond_conn_op', 'sel', 'agg', 'conds', 'order_by',
            'order_direction', 'limit', 'group_by', 'having'
        ]

    def __getitem__(self, key):
        """doc"""
        return getattr(self, key)

    def to_json(self):
        """doc"""
        return json.dumps(dict(self), ensure_ascii=False, sort_keys=True)

    def equal_all_mode(self, other):
        """doc"""
        return self.to_json() == other.to_json()

    def __eq__(self, other):
        """doc"""
        raise NotImplementedError('compare mode not set')

    def __repr__(self):
        """doc"""
        repr_str = ''
        repr_str += "sel: {}\n".format(self.sel)
        repr_str += "agg: {}\n".format([self.agg_sql_dict[a] for a in self.agg])
        repr_str += "cond_conn_op: '{}'\n".format(
            self.conn_sql_dict[self.cond_conn_op])
        repr_str += "conds: {}".format(
            [[cond[0], self.op_sql_dict[cond[1]], cond[2]]
             for cond in self.conds])

        #TODO: support order/group/...

        return repr_str

    def __str__(self):
        """doc"""
        return self.to_json()

    def _repr_html_(self):
        """doc"""
        return self.__repr__().replace('\n', '<br>')


def sql2label(sql, num_cols):
    """encode sql"""
    # because of classification task, label is from 0
    # so sel_num and cond_num should -1，and label should +1 in prediction phrase
    cond_conn_op_label = sql.cond_conn_op

    sel_num_label = sql.sel_num - 1
    # the new dataset has cond_num = 0, do not -1
    cond_num_label = len(sql.conds) + len(sql.having)
    sel_label = np.zeros(num_cols, dtype='int32')
    sel_agg_label = np.zeros((num_cols, SQL.num_agg_ops), dtype='int32')
    for col_id, agg_op in zip(sql.sel, sql.agg):
        assert col_id < num_cols, f"select col_id({col_id}) >= num_cols({num_cols}): {sql}"
        sel_agg_label[col_id][agg_op] = 1
        sel_label[col_id] = 1
    # len(SQL.op_sql_dict) over all op ID range，which means defaults to no OP
    cond_op_label = np.ones(num_cols, dtype='int32') * len(SQL.op_sql_dict)
    having_agg_label = np.zeros((num_cols, SQL.num_agg_ops), dtype='int32')

    for col_id, cond_op, _ in sql.conds:
        assert col_id < num_cols, f"where col_id({col_id}) >= num_cols({num_cols}): {sql}"
        cond_op_label[col_id] = cond_op

    for agg, col_id, cond_op, _ in sql.having:
        assert col_id < num_cols, f"having col_id({col_id}) >= num_cols({num_cols}): {sql}"
        cond_op_label[col_id] = cond_op
        having_agg_label[col_id][agg] = 1

    order_col_label = np.zeros(num_cols, dtype='int32')
    order_agg_label = np.zeros((num_cols, SQL.num_agg_ops), dtype='int32')

    order_direction_label = sql.order_direction
    for agg, order_col in sql.order_by:
        order_col_label[order_col] = 1
        order_agg_label[order_col][agg] = 1

    group_num_label = sql.group_num
    having_num_label = len(sql.having)
    group_col_label = np.zeros(num_cols, dtype='int32')
    for col_id in sql.group_by:
        assert col_id < num_cols, f"group_by col_id({col_id}) >= num_cols({num_cols}): {sql}"
        group_col_label[col_id] = 1

    return sel_num_label, cond_num_label, cond_conn_op_label, \
           sel_agg_label, sel_label, cond_op_label, \
           order_col_label, order_agg_label, order_direction_label, \
           group_num_label, having_num_label, group_col_label, having_agg_label


def decode(sel_num, sel_col, sel_agg, where_num, where_conn, where_op,
           where_op_prob, col_value, order_direction, order_col, order_agg,
           limit_label, group_num, having_num, group_col, having_agg,
           having_agg_prob, header_match_cells, candi_limit_nums):
    """decode one instance predicts to sql"""
    if col_value is None:
        col_value = [None] * len(where_op)
    # use dict to find label number, equals to label+1
    sel_num = SQL.sel_num_dict[int(sel_num)]
    sorted_sel_index = sorted(range(len(sel_col)),
                              key=lambda i: sel_col[i],
                              reverse=True)
    sel_col = [int(col_id) for col_id in sorted_sel_index][:sel_num]
    sel_agg = [int(sel_agg[col_id]) for col_id in sorted_sel_index][:sel_num]

    cond_num = SQL.cond_num_dict[int(where_num)]
    where_conn = int(where_conn)
    cond_probs = []
    conds = []
    for col_id, (cond_op, cond_prob,
                 value_id) in enumerate(zip(where_op, where_op_prob,
                                            col_value)):
        if cond_op < len(SQL.op_sql_dict):
            cond_probs.append(cond_prob)
            value = get_value_by_id(col_id, value_id, header_match_cells)
            conds.append([col_id, int(cond_op), value])
    if cond_num < len(conds):
        sorted_cond_index = sorted(range(len(cond_probs)),
                                   key=lambda i: cond_probs[i],
                                   reverse=True)
        conds = [conds[i] for i in sorted_cond_index[:cond_num]]

    if group_num is None:
        group_num = 0
    if group_num > 0:
        sorted_group_index = sorted(range(len(group_col)),
                                    key=lambda i: group_col[i],
                                    reverse=True)
        group_col = [int(col_id) for col_id in sorted_group_index[:group_num]]
    else:
        group_col = []

    having = []
    if having_num is None:
        having_num = 0
    if having_agg is not None and group_num > 0 and having_num > 0:
        having_agg_info = []
        for idx, (col_id, _, _) in enumerate(conds):
            if having_agg[col_id] > 0:
                having_agg_info.append([
                    idx,
                    int(having_agg[col_id]),
                    float(having_agg_prob[col_id])
                ])
                #cond_num -= 1
        if len(having_agg_info
               ) > 0 and having_agg_info[0][2] >= g_having_agg_threshold:
            # 按 agg 概率最大排序
            having_agg_info.sort(key=lambda x: x[2], reverse=True)
            idx, agg, _ = having_agg_info[0]
            having = [[agg] + list(conds[idx])]
            conds.pop(idx)

    order_direction = int(order_direction) if order_direction is not None else 0
    if order_direction == 0 or order_col is None or order_agg is None:
        order_by = []
        limit = '0'
    else:
        sorted_order_index = sorted(range(len(order_col)),
                                    key=lambda i: order_col[i],
                                    reverse=True)
        order_col = [int(col_id) for col_id in sorted_order_index[:1]]
        order_agg = [
            int(order_agg[col_id]) for col_id in sorted_order_index[:1]
        ]
        order_by = [[order_agg[0], order_col[0]]]
        if limit_label < len(candi_limit_nums):
            limit = candi_limit_nums[limit_label]
            if limit == '0':
                limit = '1'
        else:
            limit = '1'

    return {
        "sel": list(sel_col),
        "sel_num": int(sel_num),
        "cond_num": int(cond_num),
        "agg": list(sel_agg),
        "cond_conn_op": int(where_conn),
        "conds": [list(cond) for cond in conds],
        "order_direction": order_direction,
        "order_by": list(order_by),
        "limit": limit,
        "having_num": int(having_num),
        "group_num": int(group_num),
        "group_by": list(group_col),
        "having": list(having),
    }


def get_value_by_id(col_id, value_id, header_match_cells):
    """

    Args:
        col_id (TYPE): NULL
        value_id (TYPE): NULL
        header_match_cells (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    if value_id is None or value_id < 0:
        return None

    assert col_id < len(header_match_cells)

    curr_cells = header_match_cells[col_id]
    if len(curr_cells) == 0:
        return '0'
    if value_id >= len(curr_cells):
        return curr_cells[0]
    else:
        return curr_cells[value_id]


def decode_sqls(preds, header_lens, header_values_list, limit_nums_list):
    """Generate sqls from model outputs
    """
    fn_empty_preds = lambda: [None] * len(preds['sel_num'])

    preds_sel_num = np.argmax(preds['sel_num'], axis=-1)
    preds_sel_col = preds['sel_col']
    preds_sel_agg = np.argmax(preds['sel_agg'], axis=-1)
    preds_cond_num = np.argmax(preds['cond_num'], axis=-1)
    preds_where_conn = np.argmax(preds['where_conn'], axis=-1)
    preds_where_op = np.argmax(preds['where_op'], axis=-1)
    preds_where_op_prob = np.max(preds['where_op'], axis=-1)

    preds_order_direction = np.argmax(preds['order_direction'], axis=-1)
    preds_order_col = preds['order_col']
    preds_order_agg = np.argmax(preds['order_agg'], axis=-1)
    preds_limit_label = np.argmax(preds['limit_label'], axis=-1)

    preds_group_num = np.argmax(preds['group_num'], axis=-1)
    preds_having_num = np.argmax(preds['having_num'], axis=-1)
    preds_group_col = preds['group_col']
    preds_having_agg = np.argmax(preds['having_agg'], axis=-1)
    preds_having_agg_prob = np.max(preds['having_agg'], axis=-1)

    if g_open_value_predict:
        preds_col_value = np.argmax(preds['col_value'], axis=-1)
    else:
        preds_col_value = fn_empty_preds()

    sqls = []
    for sel_num, sel_col, sel_agg, \
            where_num, where_conn, where_op, where_op_prob, col_value, \
            order_direction, order_col, order_agg, limit_label, \
            group_num, having_num, group_col, having_agg, having_agg_prob, \
            header_len, limit_nums in zip(preds_sel_num, preds_sel_col, preds_sel_agg,
                                     preds_cond_num, preds_where_conn,
                                     preds_where_op, preds_where_op_prob, preds_col_value,
                                     preds_order_direction, preds_order_col, preds_order_agg, preds_limit_label,
                                     preds_group_num, preds_having_num,
                                     preds_group_col, preds_having_agg, preds_having_agg_prob,
                                     header_lens, limit_nums_list):

        sel_col = sel_col[:header_len]
        sel_agg = sel_agg[:header_len]
        where_op = where_op[:header_len]
        where_op_prob = where_op_prob[:header_len]
        if g_open_value_predict:
            col_value = col_value[:header_len]
        order_col = order_col[:header_len]
        order_agg = order_agg[:header_len]
        group_col = group_col[:header_len]
        having_agg = having_agg[:header_len]

        sql = decode(sel_num, sel_col, sel_agg, where_num, where_conn, where_op,
                     where_op_prob, col_value, order_direction, order_col,
                     order_agg, limit_label, group_num, having_num, group_col,
                     having_agg, having_agg_prob, None, limit_nums)
        sqls.append(sql)

    return sqls


if __name__ == "__main__":
    """run some simple test"""
    import json

    ##if len(sys.argv) > 2:
    ##    with open(sys.argv[1]) as ifs:
    ##        gold_sqls = [SQL.from_dict(json.loads(x)["sql"]) for x in ifs]
    ##    with open(sys.argv[2]) as ifs:
    ##        pred_sqls = [json.loads(x) for x in ifs]

    ##    print(f"acc of {sys.argv[1]} vs {sys.argv[2]}: ", get_acc(gold_sqls, pred_sqls))
    ##else:
    ##    gold_sqls = [{"sel": [5], "sel_num": 1, "cond_num": 2, "agg": [0],
    ##                  "cond_conn_op": 1, "conds": [[0, 2, '123'], [1, 2, '444']],
    ##                  "order_direction": "asc", "order_by": [[0, 1]]}]
    ##    pred_sqls = [{"sel": [1, 0], "agg": [0, 4], "cond_conn_op": 0,
    ##                 "conds": [], "having_conn_op": 0, "having": [], "order_by": [],
    ##                 "order_direction": "", "limit": None, "group_by": [20]}]
    ##    enc_out_names = ["sel_num_label", "cond_num_label", "cond_conn_op_label", "sel_agg_label",
    ##                     "sel_label", "cond_op_label", "order_col_label", "order_agg_label",
    ##                     "order_direction_label", "group_num_label", "group_col_label",
    ##                     "having_agg_label"]
    ##    enc_out = sql2label(SQL.from_dict(gold_sqls[0]), 8)
    ##    for name, array in zip(enc_out_names, enc_out):
    ##        print(name, array)
