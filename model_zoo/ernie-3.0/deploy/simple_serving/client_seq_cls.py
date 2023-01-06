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
import json

import requests

from paddlenlp.datasets import load_dataset

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, type=str, help="The dataset name for the simple seving")
parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for predicting.")
args = parser.parse_args()
# yapf: enable

url = "http://0.0.0.0:8189/models/ernie_cls"
headers = {"Content-Type": "application/json"}


def seq_convert_example(example):
    """convert a glue example into necessary features"""
    # Convert raw text to feature
    if "keyword" in example:  # CSL
        sentence1 = " ".join(example["keyword"])
        example = {"sentence1": sentence1, "sentence2": example["abst"], "label": example["label"]}
    elif "target" in example:  # wsc
        text, query, pronoun, query_idx, pronoun_idx = (
            example["text"],
            example["target"]["span1_text"],
            example["target"]["span2_text"],
            example["target"]["span1_index"],
            example["target"]["span2_index"],
        )
        text_list = list(text)
        assert text[pronoun_idx : (pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
        assert text[query_idx : (query_idx + len(query))] == query, "query: {}".format(query)
        if pronoun_idx > query_idx:
            text_list.insert(query_idx, "_")
            text_list.insert(query_idx + len(query) + 1, "_")
            text_list.insert(pronoun_idx + 2, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
        else:
            text_list.insert(pronoun_idx, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 1, "]")
            text_list.insert(query_idx + 2, "_")
            text_list.insert(query_idx + len(query) + 2 + 1, "_")
        text = "".join(text_list)
        example["sentence"] = text
    return example


if __name__ == "__main__":
    examples = load_dataset("clue", args.dataset)["dev"][:10]
    texts = []
    text_pairs = []
    for example in examples:
        example = seq_convert_example(example)
        if "sentence" in example:
            texts.append(example)
        else:
            texts.append(example["sentence1"])
            text_pairs.append(example["sentence2"])

    data = {
        "data": {"text": texts, "text_pair": text_pairs if len(text_pairs) > 0 else None},
        "parameters": {"max_seq_len": args.max_seq_len, "batch_size": args.batch_size},
    }
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.text)
