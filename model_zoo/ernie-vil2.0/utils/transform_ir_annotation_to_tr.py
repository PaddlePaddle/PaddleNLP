# -*- coding: utf-8 -*-

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import json

from tqdm import tqdm

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Input path of text-to-image Jsonl annotation file.")
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    t2i_record = dict()
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            text_id = obj["text_id"]
            image_ids = obj["image_ids"]
            for image_id in image_ids:
                if image_id not in t2i_record:
                    t2i_record[image_id] = []
                t2i_record[image_id].append(text_id)

    with open(args.input.replace(".jsonl", "") + ".tr.jsonl", "w", encoding="utf-8") as fout:
        for image_id, text_ids in t2i_record.items():
            out_obj = {"image_id": image_id, "text_ids": text_ids}
            fout.write("{}\n".format(json.dumps(out_obj)))

    print("Done!")
