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

import sys
import json
import numpy as np


def get_top1_from_ranker(path):
    with open(path, "r", encoding="utf-8") as f:
        scores = [float(line.strip()) for line in f.readlines()]
        top_id = np.argmax(scores)

    return top_id


def get_ocr_result_by_id(path, top_id):
    with open(path, "r", encoding="utf-8") as f:
        reses = f.readlines()
        res = reses[top_id]
    return json.loads(res)


def write_to_file(doc, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    question = sys.argv[1]
    ranker_result_path = "../Rerank/data/demo.score"
    ocr_result_path = "../OCR_process/demo_ocr_res.json"
    save_path = "data/demo_test.json"
    top_id = get_top1_from_ranker(ranker_result_path)
    doc = get_ocr_result_by_id(ocr_result_path, top_id)
    doc["question"] = question
    doc["img_id"] = str(top_id + 1)

    write_to_file(doc, save_path)
