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
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def do_convert(input_file, target_file, label_map):
    with open(input_file, 'r', encoding='utf-8') as f:
        outputs = []
        for line in f:
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
                    output.setdefault('relation_list', []).append(relation)
            outputs.append(output)

    with open(target_file, 'w', encoding='utf-8') as f:
        for output in outputs:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="duie2.0", help="The path of dataset.")
    parser.add_argument("--dataset_name", choices=['duie2.0', 'duee1.0'], type=str, default="re_data", help="The name of dataset.")

    args = parser.parse_args()
    # yapf: enable

    target_file_list = ["train_data.json", "dev_data.json"]

    if args.dataset_name == "duie2.0":
        input_file_list = ["duie_train.json", "duie_dev.json"]

        ent2id = {"DEFAULT": 0}
        rel2id = {}
        schemas = []
        schema_path = os.path.join(args.data_dir, "duie_schema.json")
        with open(schema_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_line = json.loads(line)
                subject_type = json_line['subject_type']
                obj_dict = json_line['object_type']
                predicate = json_line['predicate']
                if len(obj_dict) == 1:
                    rel2id[predicate] = len(rel2id)
                    schemas.append({
                        "object_type": list(obj_dict.keys())[0],
                        "predicate": predicate,
                        "subject_type": subject_type
                    })
                else:
                    for t in obj_dict.keys():
                        predicate_complex = predicate + "_" + t
                        rel2id[predicate_complex] = len(rel2id)
                        schemas.append({
                            "object_type": obj_dict[t],
                            "predicate": predicate_complex,
                            "subject_type": subject_type
                        })

        label_dicts = {"ent2id": ent2id, "rel2id": rel2id, "schemas": schemas}

        with open(os.path.join(args.data_dir, "label_dicts.json"),
                  "w",
                  encoding="utf-8") as fp:
            fp.write(json.dumps(label_dicts, ensure_ascii=False))

        for fi, ft in zip(input_file_list, target_file_list):
            input_file_path = os.path.join(args.data_dir, fi)
            target_file_path = os.path.join(args.data_dir, ft)
            do_convert(input_file_path, target_file_path, rel2id)
    elif args.dataset_name == "duee1.0":
        input_file_list = ["duee_train.json", "duee_dev.json"]

        schemas = []
        schema_path = os.path.join(args.data_dir, "duee_event_schema.json")
        with open(schema_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_line = json.loads(line)
                schema = {
                    "event_type": json_line["event_type"],
                    "role_list": json_line["role_list"]
                }
                schemas.append(schema)

        label_dicts = {"schemas": schemas}

        with open(os.path.join(args.data_dir, "label_dicts.json"),
                  "w",
                  encoding="utf-8") as fp:
            fp.write(json.dumps(label_dicts, ensure_ascii=False))

        for fi, ft in zip(input_file_list, target_file_list):
            input_file_path = os.path.join(args.data_dir, fi)
            target_file_path = os.path.join(args.data_dir, ft)
            os.rename(input_file_path, target_file_path)
