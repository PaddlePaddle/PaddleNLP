# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import json
import math
import random

import numpy as np
import paddle
from tqdm import tqdm

from paddlenlp.utils.log import logger

prompt_format = """你是一个阅读理解专家，请提取所给句子与问题，提取实体。请注意，如果存在实体，则一定在原句中逐字出现，请输出对应实体的原文，不要进行额外修改；如果无法提取，请输出“无相应实体”。
**句子开始**
{sentence}
**句子结束**
**问题开始**
{prompt}
**问题结束**
**回答开始**
"""


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_data_loader(dataset, mode="train", batch_size=1, trans_fn=None):
    """
    Create dataloader.
    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, return_list=True)
    return dataloader


def add_entity_negative_example(examples, texts, prompts, label_set, negative_ratio):
    negative_examples = []
    positive_examples = []
    with tqdm(total=len(prompts)) as pbar:
        for i, prompt in enumerate(prompts):
            redundants = list(set(label_set) ^ set(prompt))
            redundants.sort()

            num_positive = len(examples[i])
            if num_positive != 0:
                actual_ratio = math.ceil(len(redundants) / num_positive)
            else:
                # Set num_positive to 1 for text without positive example
                num_positive, actual_ratio = 1, 0

            if actual_ratio <= negative_ratio or negative_ratio == -1:
                idxs = [k for k in range(len(redundants))]
            else:
                idxs = random.sample(range(0, len(redundants)), negative_ratio * num_positive)

            for idx in idxs:
                src = prompt_format.format_map({"sentence": texts[i], "prompt": redundants[idx]})
                negative_result = {"src": src, "tgt": "无相应实体\n**回答结束**\n\n"}
                # negative_result = {"content": texts[i], "result_list": [], "prompt": redundants[idx]}
                negative_examples.append(negative_result)
            positive_examples.extend(examples[i])
            pbar.update(1)
    return positive_examples, negative_examples


def add_relation_negative_example(redundants, text, num_positive, ratio):
    added_example = []
    rest_example = []

    if num_positive != 0:
        actual_ratio = math.ceil(len(redundants) / num_positive)
    else:
        # Set num_positive to 1 for text without positive example
        num_positive, actual_ratio = 1, 0

    all_idxs = [k for k in range(len(redundants))]
    if actual_ratio <= ratio or ratio == -1:
        idxs = all_idxs
        rest_idxs = []
    else:
        idxs = random.sample(range(0, len(redundants)), ratio * num_positive)
        rest_idxs = list(set(all_idxs) ^ set(idxs))

    for idx in idxs:
        src = prompt_format.format_map({"sentence": text, "prompt": redundants[idx]})
        negative_result = {"src": src, "tgt": "无相应实体\n**回答结束**\n\n"}
        added_example.append(negative_result)

    for rest_idx in rest_idxs:
        src = prompt_format.format_map({"sentence": text, "prompt": redundants[idx]})
        negative_result = {"src": src, "tgt": "无相应实体\n**回答结束**\n\n"}
        rest_example.append(negative_result)

    return added_example, rest_example


def add_full_negative_example(examples, texts, relation_prompts, predicate_set, subject_goldens, schema_lang="ch"):
    with tqdm(total=len(relation_prompts)) as pbar:
        for i, relation_prompt in enumerate(relation_prompts):
            negative_sample = []
            for subject in subject_goldens[i]:
                for predicate in predicate_set:
                    # The relation prompt is constructed as follows:
                    # subject + "的" + predicate -> Chinese
                    # predicate + " of " + subject -> English
                    if schema_lang == "ch":
                        prompt = subject + "的" + predicate
                    else:
                        prompt = predicate + " of " + subject
                    if prompt not in relation_prompt:
                        src = prompt_format.format_map({"sentence": texts[i], "prompt": prompt})
                        negative_result = {"src": src, "tgt": "无相应实体\n**回答结束**\n\n"}
                        negative_sample.append(negative_result)
            examples[i].extend(negative_sample)
            pbar.update(1)
    return examples


def convert_llm_examples(
    raw_examples,
    negative_ratio,
    is_train=True,
    schema_lang="ch",
):
    """
    Convert labeled data export from doccano for extraction and aspect-level classification task.
    """

    texts = []
    entity_examples = []
    relation_examples = []
    entity_prompts = []
    relation_prompts = []
    entity_label_set = []
    entity_name_set = []
    predicate_set = []
    subject_goldens = []
    inverse_relation_list = []
    predicate_list = []

    logger.info("Converting doccano data...")
    with tqdm(total=len(raw_examples)) as pbar:
        for line in raw_examples:
            items = json.loads(line)
            # Export file in JSONL format which doccano >= 1.7.0
            # Export file in JSONL (relation) format
            # e.g. {"text": "", "relations": [ {"id": 0, "start_offset": 0, "end_offset": 6, "label": "ORG"}, ... ], "entities": [ {"id": 0, "from_id": 0, "to_id": 1, "type": "foundedAt"}, ... ]}
            text, relations, entities = items["text"], items["relations"], items["entities"]
            texts.append(text)
            entity_example = []
            entity_prompt = []
            entity_example_map = {}
            entity_map = {}  # id to entity name
            for entity in entities:
                entity_name = text[entity["start_offset"] : entity["end_offset"]]
                entity_label = entity["label"]
                entity_map[entity["id"]] = {
                    "name": entity_name,
                    "start": entity["start_offset"],
                    "end": entity["end_offset"],
                }

                src = prompt_format.format_map({"sentence": text, "prompt": entity_label})

                if entity_label not in entity_example_map.keys():
                    entity_example_map[entity_label] = {"src": src, "tgt": [entity_name]}
                else:
                    entity_example_map[entity_label]["tgt"].append(entity_name)

                if entity_label not in entity_label_set:
                    entity_label_set.append(entity_label)
                if entity_name not in entity_name_set:
                    entity_name_set.append(entity_name)
                entity_prompt.append(entity_label)

            for label, v in entity_example_map.items():
                v["tgt"] = ",".join(v["tgt"]) + "\n**回答结束**\n\n"
                entity_example.append(v)
            entity_examples.append(entity_example)
            entity_prompts.append(entity_prompt)

            subject_golden = []  # Golden entity inputs
            relation_example = []
            relation_prompt = []
            relation_example_map = {}
            inverse_relation = []
            predicates = []
            for relation in relations:
                predicate = relation["type"]
                subject_id = relation["from_id"]
                object_id = relation["to_id"]
                # The relation prompt is constructed as follows:
                # subject + "的" + predicate -> Chinese
                # predicate + " of " + subject -> English
                if schema_lang == "ch":
                    prompt = entity_map[subject_id]["name"] + "的" + predicate
                    inverse_negative = entity_map[object_id]["name"] + "的" + predicate
                else:
                    prompt = predicate + " of " + entity_map[subject_id]["name"]
                    inverse_negative = predicate + " of " + entity_map[object_id]["name"]

                if entity_map[subject_id]["name"] not in subject_golden:
                    subject_golden.append(entity_map[subject_id]["name"])

                src = prompt_format.format_map({"sentence": text, "prompt": prompt})

                inverse_relation.append(inverse_negative)
                predicates.append(predicate)

                if prompt not in relation_example_map.keys():
                    relation_example_map[prompt] = {"src": src, "tgt": [entity_map[object_id]["name"]]}
                else:
                    relation_example_map[prompt]["tgt"].append(entity_map[object_id]["name"])

                if predicate not in predicate_set:
                    predicate_set.append(predicate)
                relation_prompt.append(prompt)

            for v in relation_example_map.values():
                v["tgt"] = ",".join(v["tgt"]) + "\n**回答结束**\n\n"
                relation_example.append(v)

            relation_examples.append(relation_example)
            relation_prompts.append(relation_prompt)
            subject_goldens.append(subject_golden)
            inverse_relation_list.append(inverse_relation)
            predicate_list.append(predicates)
            pbar.update(1)

    logger.info("Adding negative samples for first stage prompt...")
    positive_examples, negative_examples = add_entity_negative_example(
        entity_examples, texts, entity_prompts, entity_label_set, negative_ratio
    )
    if len(positive_examples) == 0:
        all_entity_examples = []
    else:
        all_entity_examples = positive_examples + negative_examples

    all_relation_examples = []
    if len(predicate_set) != 0:
        logger.info("Adding negative samples for second stage prompt...")
        if is_train:

            positive_examples = []
            negative_examples = []
            per_n_ratio = negative_ratio // 3

            with tqdm(total=len(texts)) as pbar:
                for i, text in enumerate(texts):
                    negative_example = []
                    collects = []
                    num_positive = len(relation_examples[i])

                    # 1. inverse_relation_list
                    redundants1 = inverse_relation_list[i]
                    # 2. entity_name_set ^ subject_goldens[i]
                    redundants2 = []
                    if len(predicate_list[i]) != 0:
                        nonentity_list = list(set(entity_name_set) ^ set(subject_goldens[i]))
                        nonentity_list.sort()

                        if schema_lang == "ch":
                            redundants2 = [
                                nonentity + "的" + predicate_list[i][random.randrange(len(predicate_list[i]))]
                                for nonentity in nonentity_list
                            ]
                        else:
                            redundants2 = [
                                predicate_list[i][random.randrange(len(predicate_list[i]))] + " of " + nonentity
                                for nonentity in nonentity_list
                            ]
                    # 3. entity_label_set ^ entity_prompts[i]
                    redundants3 = []
                    if len(subject_goldens[i]) != 0:
                        non_ent_label_list = list(set(entity_label_set) ^ set(entity_prompts[i]))
                        non_ent_label_list.sort()

                        if schema_lang == "ch":
                            redundants3 = [
                                subject_goldens[i][random.randrange(len(subject_goldens[i]))] + "的" + non_ent_label
                                for non_ent_label in non_ent_label_list
                            ]
                        else:
                            redundants3 = [
                                non_ent_label + " of " + subject_goldens[i][random.randrange(len(subject_goldens[i]))]
                                for non_ent_label in non_ent_label_list
                            ]
                    redundants_list = [redundants1, redundants2, redundants3]

                    for redundants in redundants_list:
                        added, rest = add_relation_negative_example(
                            redundants,
                            texts[i],
                            num_positive,
                            per_n_ratio,
                        )
                        negative_example.extend(added)
                        collects.extend(rest)

                    num_sup = num_positive * negative_ratio - len(negative_example)
                    if num_sup > 0 and collects:
                        if num_sup > len(collects):
                            idxs = [k for k in range(len(collects))]
                        else:
                            idxs = random.sample(range(0, len(collects)), num_sup)
                        for idx in idxs:
                            negative_example.append(collects[idx])

                    positive_examples.extend(relation_examples[i])
                    negative_examples.extend(negative_example)
                    pbar.update(1)
            all_relation_examples = positive_examples + negative_examples
        else:
            relation_examples = add_full_negative_example(
                relation_examples, texts, relation_prompts, predicate_set, subject_goldens, schema_lang=schema_lang
            )
            all_relation_examples = [r for relation_example in relation_examples for r in relation_example]

    return all_entity_examples, all_relation_examples
