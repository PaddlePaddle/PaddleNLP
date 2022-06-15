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
import tqdm

import numpy as np
import paddle
from paddle import nn

from text2sql.models import beam_search
from text2sql.models import sql_beam_search


def inference(model,
              data,
              output_path,
              beam_size=1,
              mode='infer',
              output_history=True,
              use_heuristic=True,
              model_name='seq2tree'):
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with paddle.no_grad(), open(output_path, 'w') as ofs:
        if mode == 'infer':
            _do_infer(model, data, beam_size, output_history, ofs,
                      use_heuristic, model_name)
        elif mode == 'debug':
            _debug(model, data, ofs)


def _do_infer(model,
              data,
              beam_size,
              output_history,
              ofs,
              use_heuristic=True,
              model_name='seq2tree'):
    for i, (inputs, labels) in enumerate(tqdm.tqdm(data())):
        if model_name.startswith('seq2tree'):
            decoded = _infer_one(model, inputs, beam_size, output_history,
                                 use_heuristic, labels)
        else:
            decoded = _infer_general(model, inputs, labels)
        db_id = inputs['orig_inputs'][0].db.db_id
        question_id = inputs['orig_inputs'][0].question_id
        question = inputs['orig_inputs'][0].question
        gold_query = labels[
            0].orig_code if labels is not None and labels[0] is not None else ''
        values = inputs['orig_inputs'][0].values
        if len(decoded) == 0:
            pred_query = 'select *'
        else:
            pred_query = decoded[0]['pred_query']
        lst_output = [
            question_id,
            pred_query,
            db_id,
            json.dumps(
                {
                    'db_id': db_id,
                    'question_id': question_id,
                    'question': question,
                    'gold_query': gold_query,
                    'values': values,
                    'beams': decoded
                },
                ensure_ascii=False),
        ]
        ofs.write('\t'.join(lst_output) + '\n')
        ofs.flush()


def _infer_one(model,
               inputs,
               beam_size,
               output_history=False,
               use_heuristic=True,
               labels=None):
    """inference one example
    """
    if use_heuristic:
        # TODO: from_cond should be true from non-bert model
        beams = sql_beam_search.beam_search_with_heuristics(model,
                                                            inputs,
                                                            beam_size=beam_size,
                                                            max_steps=1000,
                                                            from_cond=False)
    else:
        beams = beam_search.beam_search(model,
                                        inputs,
                                        beam_size=beam_size,
                                        max_steps=1000)
    decoded = []
    for beam in beams:
        model_output, inferred_code = beam.inference_state.finalize()

        decoded.append({
            'pred_query':
            inferred_code,
            'model_output':
            model_output,
            'score':
            beam.score,
            **({
                'choice_history': beam.choice_history,
                'score_history': beam.score_history,
            } if output_history else {})
        })
    return decoded


def _infer_general(model, inputs, labels=None):
    output = model(inputs)
    sel_num = np.argmax(output.sel_num.numpy()).item()
    #labels[0].sel_num, labels[0].sel_col
    pred_sel_col = output.sel_col[0].numpy()
    col_ids = list(zip(range(pred_sel_col.shape[1]), pred_sel_col.tolist()[0]))
    sorted_col = sorted(col_ids, key=lambda x: x[1], reverse=True)
    pred_cols = list(sorted(sorted_col[:sel_num], key=lambda x: x[0]))
    gold_cols = []
    for cid, label in enumerate(labels[0].sel_col):
        if label == 1:
            gold_cols.append(cid)

    return {
        'sel_num': (sel_num, labels[0].sel_num),
        'sel_col': ([x[0] for x in pred_cols], gold_cols)
    }


def _debug(model, data, ofs):
    for i, item in enumerate(tqdm.tqdm(data)):
        (_, history), = model.compute_loss([item], debug=True)
        ofs.write(json.dumps({
            'index': i,
            'history': history,
        }) + '\n')
        ofs.flush()


if __name__ == "__main__":
    """run some simple test cases"""
    pass
