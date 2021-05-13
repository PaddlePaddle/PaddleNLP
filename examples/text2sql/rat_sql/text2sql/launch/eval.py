#!/usr/bin/env python3
# -*- coding:utf-8 -*-
##########################################################
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved #
##########################################################
"""do evaluating

Filname: eval.py
Authors: ZhangAo(@baidu.com)
Date: 2021-02-03 19:31:08
"""

import sys
import os
import traceback
import logging
import json

from text2sql.utils import metrics


def evaluate(model, dataset, infer_results, name='DuSQL', eval_value=True):
    """

    Args:
        model (TYPE): NULL
        dataset (TYPE): NULL
        infer_results (TYPE): NULL
        name (TYPE): Default is 'DuSQL'

    Returns: TODO

    Raises: NULL
    """
    if name.lower() == 'dusql':
        metric = metrics.MetricDuSQLAcc(dataset, eval_value=eval_value)
    else:
        raise RuntimeError(f'only supports name DuSQL. but got {name}')

    for idx, line in enumerate(infer_results):
        qid, pred_query, db_id, detail_result = line.strip().split('\t')
        dct_result = json.loads(detail_result)
        qid = dct_result['question_id']
        # data[example_id] 返回的是一个二元组 (inputs, labels)
        metric.update(dataset.get_by_qid(qid)[0], pred_query)

    eval_result = metric.finalize()
    print(
        'evaluating result:', json.dumps(
            eval_result['total_scores'], indent=4))
    with open('output/debug.json', 'w') as ofs:
        import random
        random.shuffle(eval_result['per_item'])
        json.dump(eval_result['per_item'], ofs, indent=4, ensure_ascii=False)
    return eval_result


if __name__ == "__main__":
    """run some simple test cases"""
    pass
