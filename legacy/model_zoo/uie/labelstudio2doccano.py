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

import argparse
import os
import json


def append_attrs(data, item, label_id, relation_id):

    mapp = {}

    for anno in data["annotations"][0]["result"]:
        if anno["type"] == "labels":
            label_id += 1
            item["entities"].append(
                {
                    "id": label_id,
                    "label": anno["value"]["labels"][0],
                    "start_offset": anno["value"]["start"],
                    "end_offset": anno["value"]["end"],
                }
            )
            mapp[anno["id"]] = label_id

    for anno in data["annotations"][0]["result"]:
        if anno["type"] == "relation":
            relation_id += 1
            item["relations"].append(
                {
                    "id": relation_id,
                    "from_id": mapp[anno["from_id"]],
                    "to_id": mapp[anno["to_id"]],
                    "type": anno["labels"][0],
                }
            )

    return item, label_id, relation_id


def convert(dataset, task_type):
    results = []
    outer_id = 0
    if task_type == "ext":
        label_id = 0
        relation_id = 0
        for data in dataset:
            outer_id += 1
            item = {"id": outer_id, "text": data["data"]["text"], "entities": [], "relations": []}
            item, label_id, relation_id = append_attrs(data, item, label_id, relation_id)
            results.append(item)
    # for the classification task
    else:
        for data in dataset:
            outer_id += 1
            results.append(
                {
                    "id": outer_id,
                    "text": data["data"]["text"],
                    "label": data["annotations"][0]["result"][0]["value"]["choices"],
                }
            )
    return results


def do_convert(args):

    if not os.path.exists(args.labelstudio_file):
        raise ValueError("Please input the correct path of label studio file.")

    with open(args.labelstudio_file, "r", encoding="utf-8") as infile:
        for content in infile:
            dataset = json.loads(content)
        results = convert(dataset, args.task_type)

    with open(args.doccano_file, "w", encoding="utf-8") as outfile:
        for item in results:
            outline = json.dumps(item, ensure_ascii=False)
            outfile.write(outline + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labelstudio_file", type=str, help="The export file path of label studio, only support the JSON format."
    )
    parser.add_argument("--doccano_file", type=str, default="doccano_ext.jsonl", help="Saving path in doccano format.")
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["ext", "cls"],
        default="ext",
        help="Select task type, ext for the extraction task and cls for the classification task, defaults to ext.",
    )

    args = parser.parse_args()

    do_convert(args)
