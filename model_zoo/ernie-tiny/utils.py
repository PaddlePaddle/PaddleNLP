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


def read_example(filename, intent2id, slot2id, tokenizer, max_seq_length=16, no_entity_id=0):
    """
    Reads data from file.

    tokenized_query = ['来', '一', '首', '周', '华', '健', '的', '花', '心']
    slot_sentence = '来一首<singer>周华健</singer>的<song>花心</song>'
    after processing:
    slot_label = ['O', 'O', 'O', 'B-singer', 'I-singer', 'I-singer', 'O', 'B-song', 'I-song']
    """
    for line in open(filename):
        line = line.strip().split("\t")
        if len(line) != 4:
            continue
        _, query, intent_label, slot_sentence = line
        # skip correction data
        if "||" in slot_sentence:
            continue
        tokenized_query = tokenizer.tokenize(query)
        slot_label = ["O"] * len(tokenized_query)
        query_idx = 0
        # 0 means 'O', 1 means processing in label(curr_label is accmulated)
        # 2 means copying curr_label(curr_label would not be accumulated again)
        process_id = 0
        curr_label = "O"
        for slot_char in tokenizer.tokenize(slot_sentence):
            if query_idx >= len(tokenized_query):
                break
            if slot_char == "<":
                if curr_label == "O" and process_id == 0:
                    process_id = 1
                    curr_label = "B-"
                elif process_id == 2:
                    curr_label = "O"
                continue
            if slot_char == ">":
                if "B-" in curr_label:
                    process_id = 2
                else:
                    process_id = 0
                    curr_label = "O"
                continue

            if process_id == 0:
                if slot_char == tokenized_query[query_idx]:
                    query_idx += 1
                    continue
                else:
                    curr_tokenized_query = tokenized_query[query_idx].replace("##", "")
                    if slot_char == curr_tokenized_query:
                        query_idx += 1
                        continue
                    raise ValueError("Sample error")
            elif process_id == 1:
                curr_label += slot_char
            elif process_id == 2:
                if curr_label == "O":
                    continue
                slot_label[query_idx] = curr_label
                query_idx += 1
                if "B-" in curr_label:
                    curr_label = curr_label.replace("B-", "I-")
        slot_label = [slot2id[each_slot_label] for each_slot_label in slot_label]
        tokenized_input = tokenizer(query, max_seq_len=max_seq_length, padding="max_length", truncation=True)

        example = {}
        if len(tokenized_input["input_ids"]) - 2 < len(slot_label):
            slot_label = slot_label[: len(tokenized_input["input_ids"]) - 2]

        slot_label = [no_entity_id] + slot_label + [no_entity_id]
        slot_label += [no_entity_id] * (len(tokenized_input["input_ids"]) - len(slot_label))
        example["intent_label"] = intent2id[intent_label]
        example["input_ids"] = tokenized_input["input_ids"]
        example["slot_label"] = slot_label
        yield example


def compute_metrics(p):
    intent_logits, slot_logits, padding_mask = p.predictions
    slot_preds = slot_logits.argmax(axis=-1)
    intent_preds = intent_logits.argmax(axis=-1)
    intent_label, slot_label = p.label_ids
    slot_right, intent_right = 0, 0
    for i, slot_pred in enumerate(slot_preds):
        if intent_label[i] == intent_preds[i]:
            if intent_label[i] in (0, 2, 3, 4, 6, 7, 8, 10):
                slot_right += 1
            elif ((slot_pred == slot_label[i]) | padding_mask[i]).all():
                slot_right += 1

    intent_right += sum(intent_preds == intent_label)
    accuracy = slot_right / slot_label.shape[0] * 100
    intent_accuracy = intent_right / intent_label.shape[0] * 100

    return {"accuracy": accuracy, "intent_accuracy": intent_accuracy}


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
        value.append(items)

    out_dict = {"value": value}
    return out_dict
