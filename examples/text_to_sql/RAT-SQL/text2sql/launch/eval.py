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

from text2sql.utils import metrics


def evaluate(model, dataset, infer_results, name='DuSQL', eval_value=True):
    if name.lower() == 'dusql':
        metric = metrics.MetricDuSQLAcc(dataset, eval_value=eval_value)
    else:
        raise RuntimeError(f'only supports name DuSQL. but got {name}')

    for idx, line in enumerate(infer_results):
        qid, pred_query, db_id, detail_result = line.strip().split('\t')
        dct_result = json.loads(detail_result)
        qid = dct_result['question_id']
        metric.update(dataset.get_by_qid(qid)[0], pred_query)

    eval_result = metric.finalize()
    print('evaluating result:', json.dumps(eval_result['total_scores'],
                                           indent=4))
    with open('output/debug.json', 'w') as ofs:
        import random
        random.shuffle(eval_result['per_item'])
        json.dump(eval_result['per_item'], ofs, indent=4, ensure_ascii=False)
    return eval_result


if __name__ == "__main__":
    """run some simple test cases"""
    pass
