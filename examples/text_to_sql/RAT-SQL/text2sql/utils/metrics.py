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

from text2sql.utils import nn_utils
from text2sql.utils import dusql_evaluation
from text2sql.dataproc import sql_label


class MetricSimpleSQLAcc(object):
    """SimpleSQLAccMetric. """

    def __init__(self, eval_value=False):
        """init of class

        Args:
            eval_value (TYPE): Default is False

        """
        super(MetricSimpleSQLAcc, self).__init__()

        self._eval_value = eval_value
        self._gold_list = []
        self._pred_list = []
        self._correctness = []

    def update(self, labels, predicts):
        preds = nn_utils.tensor2numpy(predicts)
        pred_sqls = sql_label.decode_sqls(preds, labels['header_lens'], None,
                                          labels['limit_nums'])
        self._gold_list.extend(labels['gold_sqls'])
        self._pred_list.extend(pred_sqls)

    def calc(self):
        conn_correct = 0
        sel_col_agg_correct = 0
        conds_correct = 0
        conds_col_correct = 0
        conds_col_op_correct = 0
        all_correct = 0
        sel_num_correct = 0
        cond_num_correct = 0
        order_correct = 0
        limit_correct = 0
        group_correct = 0
        having_correct = 0
        num_queries = len(self._gold_list)
        self._correctness.clear()
        for pred_sql, true_sql in zip(self._pred_list, self._gold_list):
            n_correct = 0
            if pred_sql['sel_num'] == true_sql.sel_num:
                sel_num_correct += 1
            if pred_sql['cond_num'] == len(true_sql.conds) + len(
                    true_sql.having):
                cond_num_correct += 1
            if pred_sql['cond_conn_op'] == true_sql.cond_conn_op:
                conn_correct += 1
                n_correct += 1
            pred_aggs = set(zip(pred_sql['sel'], pred_sql['agg']))
            true_aggs = set(zip(true_sql.sel, true_sql.agg))
            if pred_aggs == true_aggs:
                sel_col_agg_correct += 1
                n_correct += 1
            pred_conds = set([(cond[0], cond[1], cond[2])
                              for cond in pred_sql['conds']])
            if not self._eval_value:
                true_conds_tmp = [(cond[0], cond[1], None)
                                  for cond in true_sql.conds]
            else:
                true_conds_tmp = [(cond[0], cond[1], cond[2])
                                  for cond in true_sql.conds]
            true_conds = set(true_conds_tmp)
            if pred_conds == true_conds:
                conds_correct += 1
                n_correct += 1
            pred_conds_col = set([cond[0] for cond in pred_sql['conds']])
            true_conds_col = set([cond[0] for cond in true_sql['conds']])
            if pred_conds_col == true_conds_col:
                conds_col_correct += 1
            pred_conds_col_op = set([(cond[0], cond[1])
                                     for cond in pred_sql['conds']])
            true_conds_col_op = set([(cond[0], cond[1])
                                     for cond in true_sql['conds']])
            if pred_conds_col_op == true_conds_col_op:
                conds_col_op_correct += 1

            pred_order_direc = pred_sql['order_direction']
            true_order_direc = true_sql['order_direction']
            pred_order_by = pred_sql['order_by']
            true_order_by = true_sql['order_by']
            if pred_order_direc == true_order_direc and pred_order_by == true_order_by:
                n_correct += 1
                order_correct += 1

            pred_limit = pred_sql['limit']
            true_limit = true_sql['limit']
            if pred_limit == true_limit:
                n_correct += 1
                limit_correct += 1

            pred_group_by = pred_sql['group_by']
            true_group_by = true_sql['group_by']
            if pred_group_by == true_group_by:
                n_correct += 1
                group_correct += 1

            pred_having = [list(x) for x in pred_sql['having']]
            true_having = [list(x) for x in true_sql['having']]
            if not self._eval_value:
                true_having = [x[:-1] + [None] for x in true_having]
            if pred_having == true_having:
                n_correct += 1
                having_correct += 1

            if n_correct == 7:
                all_correct += 1
                self._correctness.append(1)
            else:
                self._correctness.append(0)

        self._acc = all_correct / num_queries
        self._sub_task_acc = {
            'sel_num': sel_num_correct / num_queries,
            'sel_col_agg': sel_col_agg_correct / num_queries,
            'cond_num': cond_num_correct / num_queries,
            'cond_conn': conn_correct / num_queries,
            'where_conds': conds_correct / num_queries,
            'where_col': conds_col_correct / num_queries,
            'where_col_op': conds_col_op_correct / num_queries,
        }
        self._sub_task_acc.update({
            'order_by': order_correct / num_queries,
            'limit': limit_correct / num_queries,
        })
        self._sub_task_acc.update({
            'group_by': group_correct / num_queries,
            'having': having_correct / num_queries,
        })
        return self._acc, self._sub_task_acc

    def save(self, save_dir, file_tag):
        """

        Args:
            save_dir (TYPE): NULL
            file_tag (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        if '{acc}' in file_tag or '{acc:' in file_tag:
            file_tag = file_tag.format(acc=self._acc)
        file_path = os.path.join(save_dir, file_tag)
        if not os.path.isdir(os.path.dirname(file_path)):
            os.mkdir(os.path.dirname(file_path))
        with open(file_path, 'w') as ofs:
            for pred, correct in zip(self._pred_list, self._correctness):
                pred['correct'] = correct
                ofs.write(json.dumps(pred, ensure_ascii=False) + '\n')

    def __str__(self):
        """
        Returns: TODO

        Raises: NULL
        """
        return f'acc {self._acc * 100:.2f}, sub tasks {self._sub_task_acc}'


class MetricDuSQLAcc(object):
    """Acc Metric for DuSQL like dataset"""

    def __init__(self, dataset, eval_value=True):
        """init"""
        super(MetricDuSQLAcc, self).__init__()
        self.dataset = dataset
        self.eval_value = eval_value

        self.foreign_key_maps = {
            db_id: dusql_evaluation.build_foreign_key_map(db.orig)
            for db_id, db in self.dataset.db_dict.items()
        }
        self.evaluator = dusql_evaluation.Evaluator(self.dataset.db_schema_file,
                                                    self.foreign_key_maps,
                                                    eval_value=self.eval_value)
        self.results = []

    def update(self, item, inferred_code):
        """update one instance"""
        sql_query = item.orig['query'] if 'query' in item.orig else item.orig[
            'sql_query']
        ret_dict = self.evaluator.evaluate_one(item.db.db_id, sql_query,
                                               inferred_code)
        ret_dict["db_id"] = item.orig['db_id']
        ret_dict["question"] = item.orig['question']
        self.results.append(ret_dict)

    def udpate_beams(self, item, inferred_codes, orig_question=None):
        """update one instance beam"""
        beam_dict = {}
        if orig_question:
            beam_dict["orig_question"] = orig_question
        for i, code in enumerate(inferred_codes):
            ret_dict = self.evaluator.evaluate_one(item.db.db_id,
                                                   item.orig['query'], code)
            beam_dict[i] = ret_dict
            if ret_dict["exact"] is True:
                break
        self.results.append(beam_dict)

    def finalize(self):
        """finalize"""
        self.evaluator.finalize()
        return {'per_item': self.results, 'total_scores': self.evaluator.scores}


if __name__ == "__main__":
    """run some simple test cases"""
    pass
