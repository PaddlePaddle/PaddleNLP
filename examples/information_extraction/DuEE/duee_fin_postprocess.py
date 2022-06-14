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
"""duee finance data predict post-process"""

import os
import sys
import json
import argparse

from utils import read_by_lines, write_by_lines, extract_result

enum_event_type = "公司上市"
enum_role = "环节"


def event_normalization(doc):
    """event_merge"""
    for event in doc.get("event_list", []):
        argument_list = []
        argument_set = set()
        for arg in event["arguments"]:
            arg_str = "{}-{}".format(arg["role"], arg["argument"])
            if arg_str not in argument_set:
                argument_list.append(arg)
            argument_set.add(arg_str)
        event["arguments"] = argument_list

    event_list = sorted(doc.get("event_list", []),
                        key=lambda x: len(x["arguments"]),
                        reverse=True)
    new_event_list = []
    for event in event_list:
        event_type = event["event_type"]
        event_argument_set = set()
        for arg in event["arguments"]:
            event_argument_set.add("{}-{}".format(arg["role"], arg["argument"]))
        flag = True
        for new_event in new_event_list:
            if event_type != new_event["event_type"]:
                continue
            new_event_argument_set = set()
            for arg in new_event["arguments"]:
                new_event_argument_set.add("{}-{}".format(
                    arg["role"], arg["argument"]))
            if len(event_argument_set
                   & new_event_argument_set) == len(new_event_argument_set):
                flag = False
        if flag:
            new_event_list.append(event)
    doc["event_list"] = new_event_list
    return doc


def predict_data_process(trigger_file, role_file, enum_file, schema_file,
                         save_path):
    """predict_data_process"""
    pred_ret = []
    trigger_data = read_by_lines(trigger_file)
    role_data = read_by_lines(role_file)
    enum_data = read_by_lines(enum_file)
    schema_data = read_by_lines(schema_file)
    print("trigger predict {} load from {}".format(len(trigger_data),
                                                   trigger_file))
    print("role predict {} load from {}".format(len(role_data), role_file))
    print("enum predict {} load from {}".format(len(enum_data), enum_file))
    print("schema {} load from {}".format(len(schema_data), schema_file))

    schema, sent_role_mapping, sent_enum_mapping = {}, {}, {}
    for s in schema_data:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]

    # role depends on id and sent_id
    for d in role_data:
        d_json = json.loads(d)
        r_ret = extract_result(d_json["text"], d_json["pred"]["labels"])
        role_ret = {}
        for r in r_ret:
            role_type = r["type"]
            if role_type not in role_ret:
                role_ret[role_type] = []
            role_ret[role_type].append("".join(r["text"]))
        _id = "{}\t{}".format(d_json["id"], d_json["sent_id"])
        sent_role_mapping[_id] = role_ret

    # process the enum_role data
    for d in enum_data:
        d_json = json.loads(d)
        _id = "{}\t{}".format(d_json["id"], d_json["sent_id"])
        label = d_json["pred"]["label"]
        sent_enum_mapping[_id] = label

    # process trigger data
    for d in trigger_data:
        d_json = json.loads(d)
        t_ret = extract_result(d_json["text"], d_json["pred"]["labels"])
        pred_event_types = list(set([t["type"] for t in t_ret]))
        event_list = []
        _id = "{}\t{}".format(d_json["id"], d_json["sent_id"])
        for event_type in pred_event_types:
            role_list = schema[event_type]
            arguments = []
            for role_type, ags in sent_role_mapping[_id].items():
                if role_type not in role_list:
                    continue
                for arg in ags:
                    arguments.append({"role": role_type, "argument": arg})
            # 特殊处理环节
            if event_type == enum_event_type:
                arguments.append({
                    "role": enum_role,
                    "argument": sent_enum_mapping[_id]
                })
            event = {
                "event_type": event_type,
                "arguments": arguments,
                "text": d_json["text"]
            }
            event_list.append(event)
        pred_ret.append({
            "id": d_json["id"],
            "sent_id": d_json["sent_id"],
            "text": d_json["text"],
            "event_list": event_list
        })
    doc_pred = {}
    for d in pred_ret:
        if d["id"] not in doc_pred:
            doc_pred[d["id"]] = {"id": d["id"], "event_list": []}
        doc_pred[d["id"]]["event_list"].extend(d["event_list"])

    # unfiy the all prediction results and save them
    doc_pred = [
        json.dumps(event_normalization(r), ensure_ascii=False)
        for r in doc_pred.values()
    ]
    print("submit data {} save to {}".format(len(doc_pred), save_path))
    write_by_lines(save_path, doc_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Official evaluation script for DuEE version 1.0")
    parser.add_argument("--trigger_file",
                        help="trigger model predict data path",
                        required=True)
    parser.add_argument("--role_file",
                        help="role model predict data path",
                        required=True)
    parser.add_argument("--enum_file",
                        help="enum model predict data path",
                        required=True)
    parser.add_argument("--schema_file", help="schema file path", required=True)
    parser.add_argument("--save_path", help="save file path", required=True)
    args = parser.parse_args()
    predict_data_process(args.trigger_file, args.role_file, args.enum_file,
                         args.schema_file, args.save_path)
