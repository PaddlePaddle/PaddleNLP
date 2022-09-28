# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
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

import codecs
import json
import os
import re
import zipfile

import numpy as np


def find_entity(text_raw, id_, predictions, tok_to_orig_start_index,
                tok_to_orig_end_index):
    """
    retrieval entity mention under given predicate id for certain prediction.
    this is called by the "decoding" func.
    """
    entity_list = []
    for i in range(len(predictions)):
        if [id_] in predictions[i]:
            j = 0
            while i + j + 1 < len(predictions):
                if [1] in predictions[i + j + 1]:
                    j += 1
                else:
                    break
            entity = ''.join(
                text_raw[tok_to_orig_start_index[i]:tok_to_orig_end_index[i +
                                                                          j] +
                         1])
            entity_list.append(entity)
    return list(set(entity_list))


def decoding(example_batch, id2spo, logits_batch, seq_len_batch,
             tok_to_orig_start_index_batch, tok_to_orig_end_index_batch):
    """
    model output logits -> formatted spo (as in data set file)
    """
    formatted_outputs = []
    for (i, (example, logits, seq_len, tok_to_orig_start_index, tok_to_orig_end_index)) in \
            enumerate(zip(example_batch, logits_batch, seq_len_batch, tok_to_orig_start_index_batch, tok_to_orig_end_index_batch)):

        logits = logits[1:seq_len +
                        1]  # slice between [CLS] and [SEP] to get valid logits
        logits[logits >= 0.5] = 1
        logits[logits < 0.5] = 0
        tok_to_orig_start_index = tok_to_orig_start_index[1:seq_len + 1]
        tok_to_orig_end_index = tok_to_orig_end_index[1:seq_len + 1]
        predictions = []
        for token in logits:
            predictions.append(np.argwhere(token == 1).tolist())

        # format predictions into example-style output
        formatted_instance = {}
        text_raw = example['text']
        complex_relation_label = [8, 10, 26, 32, 46]
        complex_relation_affi_label = [9, 11, 27, 28, 29, 33, 47]

        # flatten predictions then retrival all valid subject id
        flatten_predictions = []
        for layer_1 in predictions:
            for layer_2 in layer_1:
                flatten_predictions.append(layer_2[0])
        subject_id_list = []
        for cls_label in list(set(flatten_predictions)):
            if 1 < cls_label <= 56 and (cls_label + 55) in flatten_predictions:
                subject_id_list.append(cls_label)
        subject_id_list = list(set(subject_id_list))

        # fetch all valid spo by subject id
        spo_list = []
        for id_ in subject_id_list:
            if id_ in complex_relation_affi_label:
                continue  # do this in the next "else" branch
            if id_ not in complex_relation_label:
                subjects = find_entity(text_raw, id_, predictions,
                                       tok_to_orig_start_index,
                                       tok_to_orig_end_index)
                objects = find_entity(text_raw, id_ + 55, predictions,
                                      tok_to_orig_start_index,
                                      tok_to_orig_end_index)
                for subject_ in subjects:
                    for object_ in objects:
                        spo_list.append({
                            "predicate":
                            id2spo['predicate'][id_],
                            "object_type": {
                                '@value': id2spo['object_type'][id_]
                            },
                            'subject_type':
                            id2spo['subject_type'][id_],
                            "object": {
                                '@value': object_
                            },
                            "subject":
                            subject_
                        })
            else:
                #  traverse all complex relation and look through their corresponding affiliated objects
                subjects = find_entity(text_raw, id_, predictions,
                                       tok_to_orig_start_index,
                                       tok_to_orig_end_index)
                objects = find_entity(text_raw, id_ + 55, predictions,
                                      tok_to_orig_start_index,
                                      tok_to_orig_end_index)
                for subject_ in subjects:
                    for object_ in objects:
                        object_dict = {'@value': object_}
                        object_type_dict = {
                            '@value': id2spo['object_type'][id_].split('_')[0]
                        }
                        if id_ in [8, 10, 32, 46
                                   ] and id_ + 1 in subject_id_list:
                            id_affi = id_ + 1
                            object_dict[id2spo['object_type'][id_affi].split(
                                '_')[1]] = find_entity(text_raw, id_affi + 55,
                                                       predictions,
                                                       tok_to_orig_start_index,
                                                       tok_to_orig_end_index)[0]
                            object_type_dict[
                                id2spo['object_type'][id_affi].split('_')
                                [1]] = id2spo['object_type'][id_affi].split(
                                    '_')[0]
                        elif id_ == 26:
                            for id_affi in [27, 28, 29]:
                                if id_affi in subject_id_list:
                                    object_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                    find_entity(text_raw, id_affi + 55, predictions, tok_to_orig_start_index, tok_to_orig_end_index)[0]
                                    object_type_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                    id2spo['object_type'][id_affi].split('_')[0]
                        spo_list.append({
                            "predicate":
                            id2spo['predicate'][id_],
                            "object_type":
                            object_type_dict,
                            "subject_type":
                            id2spo['subject_type'][id_],
                            "object":
                            object_dict,
                            "subject":
                            subject_
                        })

        formatted_instance['text'] = example['text']
        formatted_instance['spo_list'] = spo_list
        formatted_outputs.append(formatted_instance)
    return formatted_outputs


def write_prediction_results(formatted_outputs, file_path):
    """write the prediction results"""

    with codecs.open(file_path, 'w', 'utf-8') as f:
        for formatted_instance in formatted_outputs:
            json_str = json.dumps(formatted_instance, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')
        zipfile_path = file_path + '.zip'
        f = zipfile.ZipFile(zipfile_path, 'w', zipfile.ZIP_DEFLATED)
        f.write(file_path)

    return zipfile_path


def get_precision_recall_f1(golden_file, predict_file):
    r = os.popen(
        'python3 ./re_official_evaluation.py --golden_file={} --predict_file={}'
        .format(golden_file, predict_file))
    result = r.read()
    r.close()
    precision = float(
        re.search(
            "\"precision\", \"value\":.*?}",
            result).group(0).lstrip("\"precision\", \"value\":").rstrip("}"))
    recall = float(
        re.search("\"recall\", \"value\":.*?}",
                  result).group(0).lstrip("\"recall\", \"value\":").rstrip("}"))
    f1 = float(
        re.search(
            "\"f1-score\", \"value\":.*?}",
            result).group(0).lstrip("\"f1-score\", \"value\":").rstrip("}"))

    return precision, recall, f1
