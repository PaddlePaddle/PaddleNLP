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

import numpy as np


def get_label_name(filename_intent, filename_slot):
    intent_names, slot_names = [], []
    intent2id, slot2id = {}, {}
    for id, line in enumerate(open(filename_intent)):
        line = line.strip()
        intent_names.append(line)
        intent2id[line] = id

    for id, line in enumerate(open(filename_slot)):
        line = line.strip()
        slot_names.append(line)
        slot2id[line] = id
    return intent_names, slot_names, intent2id, slot2id


# def read_example(filename, slot_names, tokenizer):
#     for line in open(filename):
#         line = line.strip().split("\t")
#         _, query, intent_label, slot_sentence = line
#         slot_label = ["O"] * len(tokenizer.tokenize(query))
#         for slot_name in slot_names:
#             splits = slot_sentence.split("<" + slot_name + ">")
#             if len(splits) > 1:
#                 pass

#         yield {"query:": query, "intent": intent_label, "slot": slot}


def read_test_file(filename):
    for line in open(filename):
        line = line.strip().split("\t")
        if len(line) < 2:
            continue
        query = line[1]
        yield {"query": query}


def input_preprocess(text, tokenizer, max_seq_length=16):
    data = tokenizer(text, max_length=max_seq_length)
    input_ids = data["input_ids"]
    return {
        "input_ids": np.array(input_ids, dtype="int32"),
    }


def intent_cls_postprocess(logits, intent_label_names):
    max_value = np.max(logits, axis=1, keepdims=True)
    exp_data = np.exp(logits - max_value)
    probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
    print(intent_label_names[int(probs.argmax(axis=-1))])
    out_dict = {"intent": intent_label_names[int(probs.argmax(axis=-1))], "confidence": probs.max(axis=-1)}
    return out_dict


def slot_cls_postprocess(logits, input_data, label_names):
    batch_preds = logits.argmax(axis=-1).tolist()
    value = []
    for batch, preds in enumerate(batch_preds):
        start = -1
        label_name = ""
        items = []
        for i, pred in enumerate(preds):
            if (label_names[pred] == "O" or "B-" in label_names[pred]) and start >= 0:
                entity = input_data[batch][start : i - 1]

                if isinstance(entity, list):
                    entity = "".join(entity)
                items.append(
                    {
                        "slot": label_name,
                        "entity": entity,
                        "pos": [start, i - 2],
                    }
                )
                start = -1
            if "B-" in label_names[pred]:
                start = i - 1
                label_name = label_names[pred][2:]
        if start >= 0:
            items.append(
                {
                    "slot": "",
                    "entity": input_data[batch][start : len(preds) - 1],
                    "pos": [start, len(preds) - 1],
                }
            )
        value.append(items)

    out_dict = {"value": value}  # , "tokens_label": batch_preds}
    return out_dict
