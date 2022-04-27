# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import json
import numpy as np
import paddle


def get_eval(tokenizer, step, data_loader, model, name):
    """
    eval test set
    """
    num_correct = 0
    num_infer = 0
    num_label = 0
    fw_gold = open(
        'output/prediction/' + name + '-gold.' + str(step),
        'w+',
        encoding='utf8')
    fw_pred = open(
        'output/prediction/' + name + '-pred.' + str(step),
        'w+',
        encoding='utf8')
    for [input_ids, token_type_ids, att_mask, pos_ids, start_ids,
         end_ids] in data_loader():
        start_prob, end_prob = model(input_ids, token_type_ids, att_mask,
                                     pos_ids)
        start_ids = paddle.cast(start_ids, 'float32')
        end_ids = paddle.cast(end_ids, 'float32')
        res = get_metric(start_prob, end_prob, start_ids, end_ids)
        num_correct += res[0]
        num_infer += res[1]
        num_label += res[2]
        get_result(tokenizer,
                   input_ids.tolist(),
                   start_ids.tolist(), end_ids.tolist(), fw_gold)
        get_result(tokenizer,
                   input_ids.tolist(),
                   start_prob.tolist(), end_prob.tolist(), fw_pred)
    fw_gold.close()
    fw_pred.close()
    res = get_f1(num_correct, num_infer, num_label)
    print('--%s --F1 %.4f --P %.4f (%i / %i) --R %.4f (%i / %i)' %
          (name, res[2], res[0], num_correct, num_infer, res[1], num_correct,
           num_label))
    return res[2]


def get_metric(start_prob, end_prob, start_ids, end_ids):
    """
    get_metric
    """
    pred_start_ids = get_bool_ids_greater_than(start_prob)
    pred_end_ids = get_bool_ids_greater_than(end_prob)
    gold_start_ids = get_bool_ids_greater_than(start_ids.tolist())
    gold_end_ids = get_bool_ids_greater_than(end_ids.tolist())

    num_correct = 0
    num_infer = 0
    num_label = 0
    for predict_start_ids, predict_end_ids, label_start_ids, label_end_ids in zip(
            pred_start_ids, pred_end_ids, gold_start_ids, gold_end_ids):
        [_correct, _infer, _label] = eval_span(
            predict_start_ids, predict_end_ids, label_start_ids, label_end_ids)
        num_correct += _correct
        num_infer += _infer
        num_label += _label
    return num_correct, num_infer, num_label


def get_f1(num_correct, num_infer, num_label):
    """
    get p r f1
    input: 10, 15, 20
    output: (0.6666666666666666, 0.5, 0.5714285714285715)
    """
    if num_infer == 0:
        precision = 0.0
    else:
        precision = num_correct * 1.0 / num_infer

    if num_label == 0:
        recall = 0.0
    else:
        recall = num_correct * 1.0 / num_label

    if num_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def get_result(tokenizer, src_ids, start_prob, end_prob, fw):
    """
    get_result
    """
    start_ids_list = get_bool_ids_greater_than(start_prob)
    end_ids_list = get_bool_ids_greater_than(end_prob)
    for start_ids, end_ids, ids in zip(start_ids_list, end_ids_list, src_ids):
        for i in reversed(range(len(ids))):
            if ids[i] != 0:
                ids = ids[:i]
                break
        span_list = get_span(start_ids, end_ids)
        src_words = " ".join(tokenizer.convert_ids_to_tokens(ids))
        span_words = [
            " ".join(tokenizer.convert_ids_to_tokens(ids[s[0]:(s[1] + 1)]))
            for s in span_list
        ]
        fw.writelines(src_words + "\n")
        fw.writelines(json.dumps(span_words, ensure_ascii=False) + "\n\n")
    return None


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    get idx of the last dim in prob arraies, which is greater than a limitation
    input: [[0.1, 0.1, 0.2, 0.5, 0.1, 0.3], [0.7, 0.6, 0.1, 0.1, 0.1, 0.1]]
        0.4
    output: [[3], [0, 1]]
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


def get_span(start_ids, end_ids, with_prob=False):
    """
    every id can only be used once
    get span set from position start and end list
    input: [1, 2, 10] [4, 12]
    output: set((2, 4), (10, 12))
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            if start_ids[start_pointer][0] == end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer][0] < end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer][0] > end_ids[end_pointer][0]:
                end_pointer += 1
                continue
        else:
            if start_ids[start_pointer] == end_ids[end_pointer]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer] < end_ids[end_pointer]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer] > end_ids[end_pointer]:
                end_pointer += 1
                continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def eval_span(predict_start_ids, predict_end_ids, label_start_ids,
              label_end_ids):
    """
    evaluate position extraction (start, end)
    return num_correct, num_infer, num_label
    input: [1, 2, 10] [4, 12] [2, 10] [4, 11]
    output: (1, 2, 2)
    """
    pred_set = get_span(predict_start_ids, predict_end_ids)
    label_set = get_span(label_start_ids, label_end_ids)
    num_correct = len(pred_set & label_set)
    num_infer = len(pred_set)
    num_label = len(label_set)
    return (num_correct, num_infer, num_label)
