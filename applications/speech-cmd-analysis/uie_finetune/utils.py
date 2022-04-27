# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import json
from tqdm import tqdm

import paddle
import random
import numpy as np


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def convert_example(example, tokenizer, max_seq_len):
    """
    example: {
        title
        prompt
        content
        result_list
    }
    """
    encoded_inputs = tokenizer(
        text=[example["prompt"]],
        text_pair=[example["content"]],
        stride=len(example["prompt"]),
        max_seq_len=max_seq_len,
        pad_to_max_seq_len=True,
        return_attention_mask=True,
        return_position_ids=True,
        return_dict=False)
    encoded_inputs = encoded_inputs[0]
    offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]
    bias = 0
    for index in range(len(offset_mapping)):
        if index == 0:
            continue
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
            bias = index
        if mapping[0] == 0 and mapping[1] == 0:
            continue
        offset_mapping[index][0] += bias
        offset_mapping[index][1] += bias
    start_ids = [0 for x in range(max_seq_len)]
    end_ids = [0 for x in range(max_seq_len)]
    for item in example["result_list"]:
        start = map_offset(item["start"] + bias, offset_mapping)
        end = map_offset(item["end"] - 1 + bias, offset_mapping)
        start_ids[start] = 1.0
        end_ids[end] = 1.0

    tokenized_output = [
        encoded_inputs["input_ids"], encoded_inputs["token_type_ids"],
        encoded_inputs["position_ids"], encoded_inputs["attention_mask"],
        start_ids, end_ids
    ]
    tokenized_output = [np.array(x, dtype="int64") for x in tokenized_output]
    return tuple(tokenized_output)


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            # print(ori_offset, index, offset_mapping)
            return index
    # print(ori_offset, -1, offset_mapping)
    return -1


def reader(data_path):
    """
    read json
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            yield json_line


def save_examples(examples, save_path, idxs):
    with open(save_path, "w", encoding="utf-8") as f:
        for idx in idxs:
            for example in examples[idx]:
                line = json.dumps(example, ensure_ascii=False) + "\n"
                f.write(line)


def add_negative_example(examples, texts, prompts, label_set, negative_ratio):
    with tqdm(total=len(prompts)) as pbar:
        for i, prompt in enumerate(prompts):
            negtive_sample = []
            redundants_list = list(set(label_set) ^ set(prompt))
            redundants_list.sort()

            if len(examples[i]) == 0:
                continue
            else:
                actual_ratio = math.ceil(
                    len(redundants_list) / len(examples[i]))

            if actual_ratio <= negative_ratio:
                idxs = [k for k in range(len(redundants_list))]
            else:
                idxs = random.sample(
                    range(0, len(redundants_list)),
                    negative_ratio * len(examples[i]))

            for idx in idxs:
                negtive_result = {
                    "content": texts[i],
                    "result_list": [],
                    "prompt": redundants_list[idx]
                }
                negtive_sample.append(negtive_result)
            examples[i].extend(negtive_sample)
            pbar.update(1)
    return examples


def construct_relation_label_set(entity_name_set, predicate_set):
    relation_label_set = set()
    for entity_name in entity_name_set:
        for predicate in predicate_set:
            relation_label = entity_name + "的" + predicate
            relation_label_set.add(relation_label)
    return sorted(list(relation_label_set))


def convert_doccano_examples(raw_examples, negative_ratio):
    texts = []
    entity_examples = []
    relation_examples = []
    entity_prompts = []
    relation_prompts = []
    entity_label_set = []
    entity_name_set = []
    predicate_set = []

    print(f"Converting doccano data...")
    with tqdm(total=len(raw_examples)) as pbar:
        for line in raw_examples:
            items = json.loads(line)
            text, relations, entities = items["text"], items[
                "relations"], items["entities"]
            texts.append(text)

            entity_example = []
            entity_prompt = []
            entity_example_map = {}
            entity_map = {}  # id to entity name
            for entity in entities:
                entity_name = text[entity["start_offset"]:entity["end_offset"]]
                entity_map[entity["id"]] = {
                    "name": entity_name,
                    "start": entity["start_offset"],
                    "end": entity["end_offset"]
                }

                entity_label = entity["label"]
                result = {
                    "text": entity_name,
                    "start": entity["start_offset"],
                    "end": entity["end_offset"]
                }
                if entity_label not in entity_example_map.keys():
                    entity_example_map[entity_label] = {
                        "content": text,
                        "result_list": [result],
                        "prompt": entity_label
                    }
                else:
                    entity_example_map[entity_label]["result_list"].append(
                        result)

                if entity_label not in entity_label_set:
                    entity_label_set.append(entity_label)
                if entity_name not in entity_name_set:
                    entity_name_set.append(entity_name)
                entity_prompt.append(entity_label)

            for v in entity_example_map.values():
                entity_example.append(v)

            entity_examples.append(entity_example)
            entity_prompts.append(entity_prompt)

            relation_example = []
            relation_prompt = []
            relation_example_map = {}
            for relation in relations:
                predicate = relation["type"]
                subject_id = relation["from_id"]
                object_id = relation["to_id"]
                relation_label = entity_map[subject_id][
                    "name"] + "的" + predicate
                result = {
                    "text": entity_map[object_id]["name"],
                    "start": entity_map[object_id]["start"],
                    "end": entity_map[object_id]["end"]
                }
                if relation_label not in relation_example_map.keys():
                    relation_example_map[relation_label] = {
                        "content": text,
                        "result_list": [result],
                        "prompt": relation_label
                    }
                else:
                    relation_example_map[relation_label]["result_list"].append(
                        result)

                if predicate not in predicate_set:
                    predicate_set.append(predicate)
                relation_prompt.append(relation_label)

            for v in relation_example_map.values():
                relation_example.append(v)

            relation_examples.append(relation_example)
            relation_prompts.append(relation_prompt)
            pbar.update(1)

    print(f"Adding negative samples for first stage prompt...")
    entity_examples = add_negative_example(entity_examples, texts,
                                           entity_prompts, entity_label_set,
                                           negative_ratio)

    print(f"Constructing relation labels...")
    relation_label_set = construct_relation_label_set(entity_name_set,
                                                      predicate_set)

    print(f"Adding negative samples for second stage prompt...")
    relation_examples = add_negative_example(relation_examples, texts,
                                             relation_prompts,
                                             relation_label_set, negative_ratio)
    return entity_examples, relation_examples


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
