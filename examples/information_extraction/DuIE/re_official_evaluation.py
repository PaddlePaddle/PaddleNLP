# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# imitations under the License.
"""
This module to calculate precision, recall and f1-value 
of the predicated results.
"""
import sys
import json
import os
import zipfile
import traceback
import argparse

SUCCESS = 0
FILE_ERROR = 1
NOT_ZIP_FILE = 2
ENCODING_ERROR = 3
JSON_ERROR = 4
SCHEMA_ERROR = 5
ALIAS_FORMAT_ERROR = 6

CODE_INFO = {
    SUCCESS: 'success',
    FILE_ERROR: 'file is not exists',
    NOT_ZIP_FILE: 'predict file is not a zipfile',
    ENCODING_ERROR: 'file encoding error',
    JSON_ERROR: 'json parse is error',
    SCHEMA_ERROR: 'schema is error',
    ALIAS_FORMAT_ERROR: 'alias dict format is error'
}


def del_bookname(entity_name):
    """delete the book name"""
    if entity_name.startswith(u'《') and entity_name.endswith(u'》'):
        entity_name = entity_name[1:-1]
    return entity_name


def check_format(line):
    """检查输入行是否格式错误"""
    ret_code = SUCCESS
    json_info = {}
    try:
        line = line.strip()
    except:
        ret_code = ENCODING_ERROR
        return ret_code, json_info
    try:
        json_info = json.loads(line)
    except:
        ret_code = JSON_ERROR
        return ret_code, json_info
    if 'text' not in json_info or 'spo_list' not in json_info:
        ret_code = SCHEMA_ERROR
        return ret_code, json_info
    required_key_list = ['subject', 'predicate', 'object']
    for spo_item in json_info['spo_list']:
        if type(spo_item) is not dict:
            ret_code = SCHEMA_ERROR
            return ret_code, json_info
        if not all(
            [required_key in spo_item for required_key in required_key_list]):
            ret_code = SCHEMA_ERROR
            return ret_code, json_info
        if not isinstance(spo_item['subject'], str) or \
                not isinstance(spo_item['object'], dict):
            ret_code = SCHEMA_ERROR
            return ret_code, json_info
    return ret_code, json_info


def _parse_structured_ovalue(json_info):
    spo_result = []
    for item in json_info["spo_list"]:
        s = del_bookname(item['subject'].lower())
        o = {}
        for o_key, o_value in item['object'].items():
            o_value = del_bookname(o_value).lower()
            o[o_key] = o_value
        spo_result.append({"predicate": item['predicate'], \
                           "subject": s, \
                           "object": o})
    return spo_result


def load_predict_result(predict_filename):
    """Loads the file to be predicted"""
    predict_result = {}
    ret_code = SUCCESS
    if not os.path.exists(predict_filename):
        ret_code = FILE_ERROR
        return ret_code, predict_result
    try:
        predict_file_zip = zipfile.ZipFile(predict_filename)
    except:
        ret_code = NOT_ZIP_FILE
        return ret_code, predict_result
    for predict_file in predict_file_zip.namelist():
        for line in predict_file_zip.open(predict_file):
            ret_code, json_info = check_format(line)
            if ret_code != SUCCESS:
                return ret_code, predict_result
            sent = json_info['text']
            spo_result = _parse_structured_ovalue(json_info)
            predict_result[sent] = spo_result
    return ret_code, predict_result


def load_test_dataset(golden_filename):
    """load golden file"""
    golden_dict = {}
    ret_code = SUCCESS
    if not os.path.exists(golden_filename):
        ret_code = FILE_ERROR
        return ret_code, golden_dict
    with open(golden_filename, 'r', encoding="utf-8") as gf:
        for line in gf:
            ret_code, json_info = check_format(line)
            if ret_code != SUCCESS:
                return ret_code, golden_dict

            sent = json_info['text']
            spo_result = _parse_structured_ovalue(json_info)
            golden_dict[sent] = spo_result
    return ret_code, golden_dict


def load_alias_dict(alias_filename):
    """load alias dict"""
    alias_dict = {}
    ret_code = SUCCESS
    if alias_filename == "":
        return ret_code, alias_dict
    if not os.path.exists(alias_filename):
        ret_code = FILE_ERROR
        return ret_code, alias_dict
    with open(alias_filename, "r", encoding="utf-8") as af:
        for line in af:
            line = line.strip()
            try:
                words = line.split('\t')
                alias_dict[words[0].lower()] = set()
                for alias_word in words[1:]:
                    alias_dict[words[0].lower()].add(alias_word.lower())
            except:
                ret_code = ALIAS_FORMAT_ERROR
                return ret_code, alias_dict
    return ret_code, alias_dict


def del_duplicate(spo_list, alias_dict):
    """delete synonyms triples in predict result"""
    normalized_spo_list = []
    for spo in spo_list:
        if not is_spo_in_list(spo, normalized_spo_list, alias_dict):
            normalized_spo_list.append(spo)
    return normalized_spo_list


def is_spo_in_list(target_spo, golden_spo_list, alias_dict):
    """target spo是否在golden_spo_list中"""
    if target_spo in golden_spo_list:
        return True
    target_s = target_spo["subject"]
    target_p = target_spo["predicate"]
    target_o = target_spo["object"]
    target_s_alias_set = alias_dict.get(target_s, set())
    target_s_alias_set.add(target_s)
    for spo in golden_spo_list:
        s = spo["subject"]
        p = spo["predicate"]
        o = spo["object"]
        if p != target_p:
            continue
        if s in target_s_alias_set and _is_equal_o(o, target_o, alias_dict):
            return True
    return False


def _is_equal_o(o_a, o_b, alias_dict):
    for key_a, value_a in o_a.items():
        if key_a not in o_b:
            return False
        value_a_alias_set = alias_dict.get(value_a, set())
        value_a_alias_set.add(value_a)
        if o_b[key_a] not in value_a_alias_set:
            return False
    for key_b, value_b in o_b.items():
        if key_b not in o_a:
            return False
        value_b_alias_set = alias_dict.get(value_b, set())
        value_b_alias_set.add(value_b)
        if o_a[key_b] not in value_b_alias_set:
            return False
    return True


def calc_pr(predict_filename, alias_filename, golden_filename):
    """calculate precision, recall, f1"""
    ret_info = {}

    #load alias dict
    ret_code, alias_dict = load_alias_dict(alias_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        return ret_info
    #load test golden dataset
    ret_code, golden_dict = load_test_dataset(golden_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        return ret_info
    #load predict result
    ret_code, predict_result = load_predict_result(predict_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        return ret_info

    #evaluation
    correct_sum, predict_sum, recall_sum, recall_correct_sum = 0.0, 0.0, 0.0, 0.0
    for sent in golden_dict:
        golden_spo_list = del_duplicate(golden_dict[sent], alias_dict)
        predict_spo_list = predict_result.get(sent, list())
        normalized_predict_spo = del_duplicate(predict_spo_list, alias_dict)
        recall_sum += len(golden_spo_list)
        predict_sum += len(normalized_predict_spo)
        for spo in normalized_predict_spo:
            if is_spo_in_list(spo, golden_spo_list, alias_dict):
                correct_sum += 1
        for golden_spo in golden_spo_list:
            if is_spo_in_list(golden_spo, predict_spo_list, alias_dict):
                recall_correct_sum += 1
    sys.stderr.write('correct spo num = {}\n'.format(correct_sum))
    sys.stderr.write('submitted spo num = {}\n'.format(predict_sum))
    sys.stderr.write('golden set spo num = {}\n'.format(recall_sum))
    sys.stderr.write(
        'submitted recall spo num = {}\n'.format(recall_correct_sum))
    precision = correct_sum / predict_sum if predict_sum > 0 else 0.0
    recall = recall_correct_sum / recall_sum if recall_sum > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
            if precision + recall > 0 else 0.0
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    ret_info['errorCode'] = SUCCESS
    ret_info['errorMsg'] = CODE_INFO[SUCCESS]
    ret_info['data'] = []
    ret_info['data'].append({'name': 'precision', 'value': precision})
    ret_info['data'].append({'name': 'recall', 'value': recall})
    ret_info['data'].append({'name': 'f1-score', 'value': f1})
    return ret_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden_file",
                        type=str,
                        help="true spo results",
                        required=True)
    parser.add_argument("--predict_file",
                        type=str,
                        help="spo results predicted",
                        required=True)
    parser.add_argument("--alias_file",
                        type=str,
                        default='',
                        help="entities alias dictionary")
    args = parser.parse_args()
    golden_filename = args.golden_file
    predict_filename = args.predict_file
    alias_filename = args.alias_file
    ret_info = calc_pr(predict_filename, alias_filename, golden_filename)
    print(json.dumps(ret_info))
