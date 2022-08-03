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

import re
import os
import json
import argparse


def search(pattern, sequence):
    """Find substrings"""
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def do_convert(input_file,
               target_file,
               label_map,
               task_type,
               dataset_name=None):
    outputs = []

    if task_type == "relation_extraction":
        with open(input_file, 'r', encoding='utf-8') as fp:
            for line in fp:
                output = {}
                json_line = json.loads(line)
                text = json_line['text']
                output['text'] = text
                spo_list = json_line['spo_list']
                for spo in spo_list:
                    subject_text = spo['subject']
                    if len(subject_text) == 0:
                        continue
                    predicate_text = spo['predicate']
                    subj_start_id = search(subject_text, text)
                    entity = {
                        'text': subject_text,
                        'type': "DEFAULT",
                        'start_index': subj_start_id
                    }
                    output.setdefault('entity_list', []).append(entity)
                    for spo_object in spo['object'].keys():
                        object_text = spo['object'][spo_object]
                        if len(object_text) == 0:
                            continue
                        obj_start_id = search(object_text, text)
                        entity = {
                            'text': object_text,
                            'type': "DEFAULT",
                            'start_index': obj_start_id
                        }
                        output.setdefault('entity_list', []).append(entity)
                        if predicate_text in label_map.keys():
                            # Simple relation
                            relation = {
                                'subject': subject_text,
                                'predicate': predicate_text,
                                'object': object_text,
                                'subject_start_index': subj_start_id,
                                'object_start_index': obj_start_id
                            }
                        else:
                            relation = {
                                'subject': subject_text,
                                'predicate': predicate_text + '_' + spo_object,
                                'object': object_text,
                                'subject_start_index': subj_start_id,
                                'object_start_index': obj_start_id
                            }
                        output.setdefault('spo_list', []).append(relation)
                outputs.append(output)
    elif task_type == "entity_extraction":
        if dataset_name == "CMeEE":
            data = json.load(open(input_file))
            for sample in data:
                output = {"text": sample['text'], "spo_list": []}
                for e in sample["entities"]:
                    entity = {
                        "text": e['entity'],
                        "type": e['type'],
                        "start_index": e['start_idx']
                    }
                    output.setdefault('entity_list', []).append(entity)
                outputs.append(output)
        elif dataset_name == "CLUENER":
            with open(input_file, 'r', encoding='utf-8') as fp:
                for line in fp:
                    json_line = json.loads(line)
                    text = json_line['text']
                    output = {'text': text}
                    labels = json_line['label']
                    for key1 in labels.keys():
                        for key2 in labels[key1].keys():
                            for position in labels[key1][key2]:
                                entity = {
                                    "text": key2,
                                    "type": key1,
                                    "start_index": position[0]
                                }
                                output.setdefault('entity_list',
                                                  []).append(entity)
                    outputs.append(output)

    with open(target_file, 'w', encoding='utf-8') as fp:
        for output in outputs:
            fp.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="ner_data", help="The path of dataset.")
    parser.add_argument("--dataset_name", choices=['DuIE2.0', 'DuEE1.0', "CMeEE", "CLUENER"], type=str, default="CLUENER", help="The name of dataset.")

    args = parser.parse_args()
    # yapf: enable

    # The converted file is uniformly named to `train_data.json` and `dev_data.json`
    target_file_list = ["train_data.json", "dev_data.json"]

    if args.dataset_name == "DuIE2.0":
        input_file_list = ["duie_train.json", "duie_dev.json"]

        # The entity type is default to `DEFAULT` if only the SPO Triplet is extracted
        entity2id = {"DEFAULT": 0}
        relation2id = {}
        schema_path = os.path.join(args.data_dir, "duie_schema.json")
        with open(schema_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                json_line = json.loads(line)
                subject_type = json_line['subject_type']
                obj_dict = json_line['object_type']
                predicate = json_line['predicate']
                if len(obj_dict) == 1:
                    relation2id[predicate] = len(relation2id)
                else:
                    for t in obj_dict.keys():
                        predicate_complex = predicate + "_" + t
                        relation2id[predicate_complex] = len(relation2id)

        label_dict = {
            "entity2id": entity2id,
            "relation2id": relation2id,
        }

        with open(os.path.join(args.data_dir, "label_dict.json"),
                  "w",
                  encoding="utf-8") as fp:
            fp.write(json.dumps(label_dict, ensure_ascii=False))

        for fi, ft in zip(input_file_list, target_file_list):
            input_file_path = os.path.join(args.data_dir, fi)
            target_file_path = os.path.join(args.data_dir, ft)
            do_convert(input_file_path, target_file_path, relation2id,
                       "relation_extraction")
    elif args.dataset_name == "DuEE1.0":
        input_file_list = ["duee_train.json", "duee_dev.json"]

        schemas = []
        schema_path = os.path.join(args.data_dir, "duee_event_schema.json")
        with open(schema_path, 'r', encoding='utf-8') as fp:
            for line in f:
                json_line = json.loads(line)
                schema = {
                    "event_type": json_line["event_type"],
                    "role_list": json_line["role_list"]
                }
                schemas.append(schema)

        label_dict = {"schema_list": schemas}

        with open(os.path.join(args.data_dir, "label_dict.json"),
                  "w",
                  encoding="utf-8") as fp:
            fp.write(json.dumps(label_dict, ensure_ascii=False))

        for fi, ft in zip(input_file_list, target_file_list):
            input_file_path = os.path.join(args.data_dir, fi)
            target_file_path = os.path.join(args.data_dir, ft)
            os.rename(input_file_path, target_file_path)
    elif args.dataset_name == "CMeEE":
        entity2id = {
            "dis": 0,
            "sym": 1,
            "dru": 2,
            "equ": 3,
            "pro": 4,
            "bod": 5,
            "ite": 6,
            "mic": 7,
            "dep": 8
        }
        input_file_list = ["CMeEE_train.json", "CMeEE_dev.json"]

        label_dict = {"entity2id": entity2id}

        with open(os.path.join(args.data_dir, "label_dict.json"),
                  "w",
                  encoding="utf-8") as fp:
            fp.write(json.dumps(label_dict, ensure_ascii=False))

        for fi, ft in zip(input_file_list, target_file_list):
            input_file_path = os.path.join(args.data_dir, fi)
            target_file_path = os.path.join(args.data_dir, ft)
            do_convert(input_file_path, target_file_path, entity2id,
                       "entity_extraction", "CMeEE")
    elif args.dataset_name == "CLUENER":
        entity2id = {
            "address": 0,
            "book": 1,
            "company": 2,
            "game": 3,
            "government": 4,
            "movie": 5,
            "name": 6,
            "organization": 7,
            "position": 8,
            "scene": 9
        }
        input_file_list = ["train.json", "dev.json"]

        label_dict = {"entity2id": entity2id}

        with open(os.path.join(args.data_dir, "label_dict.json"),
                  "w",
                  encoding="utf-8") as fp:
            fp.write(json.dumps(label_dict, ensure_ascii=False))
        for fi, ft in zip(input_file_list, target_file_list):
            input_file_path = os.path.join(args.data_dir, fi)
            target_file_path = os.path.join(args.data_dir, ft)
            do_convert(input_file_path, target_file_path, entity2id,
                       "entity_extraction", "CLUENER")
