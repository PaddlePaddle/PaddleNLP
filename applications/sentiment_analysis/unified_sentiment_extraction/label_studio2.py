# coding=utf-8
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

import os
import copy
import time
import math
import argparse
import json
import random
from tqdm import tqdm
from decimal import Decimal

import numpy as np
import paddle
from paddlenlp.utils.log import logger

from utils import load_txt


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class Convertor(object):
    """Convertor to convert data export from annotation platform"""

    def __init__(self,
                 negative_ratio=5,
                 prompt_prefix="情感倾向",
                 options=["正向", "负向"],
                 separator="##",
                 default_relation_name="观点词"):
        """Init Data Convertor"""
        self.negative_ratio = negative_ratio
        self.prompt_prefix = prompt_prefix
        self.options = options
        self.separator = separator
        self.default_relation_name = default_relation_name

    def process_text_tag(self, line, task_type="ext"):
        items = {}
        items['text'] = line['data']['text']
        if task_type == "ext":
            items['entities'] = []
            items['relations'] = []
            items["relation_ids"] = set()
            result_list = line['annotations'][0]['result']
            for a in result_list:
                if a['type'] == "labels":
                    items['entities'].append({
                        "id": a['id'],
                        "start_offset": a['value']['start'],
                        "end_offset": a['value']['end'],
                        "label": a['value']['labels'][0]
                    })
                else:
                    items['relations'].append({
                        "id":
                        a['from_id'] + "-" + a['to_id'],
                        "from_id":
                        a['from_id'],
                        "to_id":
                        a['to_id'],
                        "type":
                        # modify
                        a['labels'][0] if a['labels'] else self.default_relation_name
                    })
                    items["relation_ids"].add(a['from_id'])
                    items["relation_ids"].add(a['to_id'])

        elif task_type == "cls":
            items['label'] = line['annotations'][0]['result'][0]['value'][
                'choices']
        return items

    def convert_cls_examples(self, raw_examples):
        """
        Convert labeled data for classification task.
        """
        examples = []
        logger.info(f"Converting annotation data...")
        with tqdm(total=len(raw_examples)) as pbar:
            for line in raw_examples:
                items = self.process_text_tag(line, task_type="cls")
                text, labels = items["text"], items["label"]
                example = self.generate_cls_example(text, labels,
                                                    self.prompt_prefix,
                                                    self.options)
                examples.append(example)
        return examples

    def convert_ext_examples(self, raw_examples, is_train=True, synonyms=None, implicit_opinion_map=None):
        """
        Convert labeled data for extraction task.
        """

        def _sep_cls_label(label, separator):
            label_list = label.split(separator)
            if len(label_list) == 1:
                return label_list[0], None
            return label_list[0], label_list[1:]

        texts = []
        # {"content": "", "result_list": [], "prompt": "X"}
        entity_examples = []
        # {"content": "", "result_list": [], "prompt": "X的Y"}
        relation_examples = []
        # {"content": "", "result_list": [], "prompt": "X的情感倾向[正向，负向]"}
        entity_cls_examples = []
        # same to relation_examples
        implicit_examples = []

        # Entity label set: ["时间", "地点", ... ]
        entity_label_set = []
        # Entity name set: ["2月8日上午", "北京", ... ]
        entity_name_set = []
        # Predicate set: ["歌手", "所属专辑", ... ]
        predicate_set = []
        # Set of subject entity in relation: ["房间", "价格", ... ]
        subject_name_set = []

        # List[List[str]]
        # List of entity prompt for each example
        entity_prompt_list = []
        # List of relation prompt for each example
        relation_prompt_list = []
        # Golden subject label for each example
        subject_golden_list = []
        # List of inverse relation for each example
        inverse_relation_list = []
        # List of predicate for each example
        predicate_list = []

        logger.info(f"Converting annotation data...")
        with tqdm(total=len(raw_examples)) as pbar:
            for line in raw_examples:
                items = self.process_text_tag(line, task_type="ext")

                text, relations, entities, relation_ids = items["text"], items[
                    "relations"], items["entities"], items["relation_ids"]
                texts.append(text)

                entity_example = []
                entity_prompt = []
                entity_example_map = {}
                entity_map = {}  # id to entity name
                implict_dict = {} # implicit aspect to opinions
                for entity in entities:
                    entity_name = text[
                        entity["start_offset"]:entity["end_offset"]]
                    entity_map[entity["id"]] = {
                        "name": entity_name,
                        "start": entity["start_offset"],
                        "end": entity["end_offset"]
                    }

                    entity_label, entity_cls_label = _sep_cls_label(
                        entity["label"], self.separator)

                    # generate examples for entity-level sentiment classification
                    # xxx + "的" + 情感倾向
                    if entity_cls_label is not None:
                        entity_cls_prompt_prefix = entity_name + "的" + self.prompt_prefix
                        entity_cls_example = self.generate_cls_example(
                            text, entity_cls_label, entity_cls_prompt_prefix, self.options)

                        entity_cls_examples.append(entity_cls_example)

                    # generate examples for entity extraction
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

                    if implicit_opinion_map and entity["id"] not in relation_ids:
                        maped_entity = entity_map[entity["id"]]
                        if maped_entity["name"] not in implicit_opinion_map:
                            continue
                        result = {"text":maped_entity["name"], "start":maped_entity["start"], "end":maped_entity["end"]}
                        aspect = implicit_opinion_map[maped_entity["name"]]
                        if aspect not in implict_dict:
                            implict_dict[aspect] = [result]
                        else:
                            implict_dict[aspect].append(result)

                for v in entity_example_map.values():
                    entity_example.append(v)
                entity_examples.append(entity_example)
                entity_prompt_list.append(entity_prompt)

                # generate examples for extraction of implicit opinion
                for entity_name in implict_dict.keys():
                    prompt = entity_name + "的" + self.default_relation_name
                    implicit_example = {
                        "content": text,
                        "result_list": implict_dict[entity_name],
                        "prompt": prompt
                    }
                    implicit_examples.append(implicit_example)

                # generate examples for relation extraction
                # Golden entity inputs
                subject_golden = [] 
                relation_example = []
                relation_prompt = []
                relation_example_map = {}
                inverse_relation = []
                predicates = []
                for relation in relations:
                    predicate = relation["type"]
                    subject_id = relation["from_id"]
                    object_id = relation["to_id"]

                    prompt = entity_map[subject_id]["name"] + "的" + predicate
                    inverse_negative = entity_map[object_id]["name"] + "的" + predicate

                    if entity_map[subject_id]["name"] not in subject_golden:
                        subject_golden.append(entity_map[subject_id]["name"])
                    if entity_map[subject_id]["name"] not in subject_name_set:
                        subject_name_set.append(entity_map[subject_id]["name"])

                    result = {
                        "text": entity_map[object_id]["name"],
                        "start": entity_map[object_id]["start"],
                        "end": entity_map[object_id]["end"]
                    }

                    inverse_relation.append(inverse_negative)
                    predicates.append(predicate)

                    if prompt not in relation_example_map.keys():
                        relation_example_map[prompt] = {
                            "content": text,
                            "result_list": [result],
                            "prompt": prompt
                        }
                    else: 
                        relation_example_map[prompt]["result_list"].append(
                            result)

                    if predicate not in predicate_set:
                        predicate_set.append(predicate)
                    relation_prompt.append(prompt)

                for v in relation_example_map.values():
                    relation_example.append(v)

                relation_examples.append(relation_example)
                relation_prompt_list.append(relation_prompt)
                subject_golden_list.append(subject_golden)
                inverse_relation_list.append(inverse_relation)
                predicate_list.append(predicates)
                pbar.update(1)

        # generate negative examples according to entity
        logger.info(f"Adding negative examples for first stage prompt...")
        positive_examples, negative_examples = self.add_entity_negative_example(
            entity_examples, texts, entity_prompt_list, entity_label_set)
        if len(positive_examples) == 0:
            all_entity_examples = []
        else:
            all_entity_examples = positive_examples + negative_examples

        # generate negative examples according to relation
        all_relation_examples = []
        if len(predicate_set) != 0:
            logger.info(f"Adding negative examples for second stage prompt...")
            if is_train:
                positive_examples = []
                negative_examples = []
                per_n_ratio = self.negative_ratio // 3

                with tqdm(total=len(texts)) as pbar:
                    for i, text in enumerate(texts):
                        negative_example = []
                        collects = []
                        # num_positive = len(relation_examples[i])

                        # 1. inverse_relation_list
                        # 热情(观点词)的观点词
                        redundants1 = inverse_relation_list[i]

                        # 2. subject_name_set ^ subject_golden_list[i]
                        # 汽车(该属性不在当前样本中)的观点词
                        redundants2 = []
                        if len(predicate_list[i]) != 0:
                            nonentity_list = list(
                                set(subject_name_set)
                                ^ set(subject_golden_list[i]))
                            nonentity_list.sort()

                            redundants2 = [
                                nonentity + "的" +
                                predicate_list[i][random.randrange(
                                    len(predicate_list[i]))]
                                for nonentity in nonentity_list
                            ]

                        # 3. entity_label_set ^ entity_prompt_list[i]
                        # 房间的[^评价维度，观点词]
                        redundants3 = []
                        if len(subject_golden_list[i]) != 0:
                            non_ent_label_list = list(
                                set(entity_label_set)
                                ^ set(entity_prompt_list[i]))
                            non_ent_label_list.sort()

                            redundants3 = [
                                subject_golden_list[i][random.randrange(
                                    len(subject_golden_list[i]))] + "的" +
                                non_ent_label
                                for non_ent_label in non_ent_label_list
                            ]

                        redundants_list = [
                            redundants1, redundants2, redundants3
                        ]

                        for redundants in redundants_list:
                            added, rest = self.add_relation_negative_example(
                                redundants,
                                texts[i],
                                per_n_ratio
                            )
                            negative_example.extend(added)
                            collects.extend(rest)
                        num_sup = self.negative_ratio - len(negative_example)
                        if num_sup > 0 and collects:
                            if num_sup > len(collects):
                                idxs = [k for k in range(len(collects))]
                            else:
                                idxs = random.sample(range(0, len(collects)),
                                                     num_sup)
                            for idx in idxs:
                                negative_example.append(collects[idx])
                        positive_examples.extend(relation_examples[i])
                        negative_examples.extend(negative_example)
                        
                        pbar.update(1)
                all_relation_examples = positive_examples + negative_examples

            else:
                relation_examples = self.add_full_negative_example(
                    relation_examples, texts, relation_prompt_list,
                    predicate_set, subject_golden_list)
                all_relation_examples = [
                    r for relation_example in relation_examples
                    for r in relation_example
                ]

        # generate negative examples according to sentiment polarity
        all_cls_examples = entity_cls_examples
        if len(self.options) == 3 and "未提及" in self.options:
            logger.info(f"Adding negative examples for third stage prompt...")
            if is_train:
                cls_negatives_examples = self.add_cls_negative_example(texts, subject_name_set, subject_golden_list)
            else:
                cls_negatives_examples = self.add_cls_negative_example(texts, subject_name_set, subject_golden_list, with_full=True)
            all_cls_examples += cls_negatives_examples

        # add implicit examples to relation examples list
        if implicit_examples:
            logger.info(f"Adding  examples with implicit opinions...")
            all_relation_examples += implicit_examples

        # generate examples with synonyms to support aspect aggregation
        if synonyms is not None:
            logger.info(f"Expand examples with synonyms...")
            synonym_map = {}
            for k, vs in synonyms.items():
                for v in vs:
                    synonym_map[v] = k

            relation_synonym_examples = self.change_aspect_with_synonyms(all_relation_examples, synonyms, synonym_map)
            all_relation_examples += relation_synonym_examples
            cls_synonym_examples = self.change_aspect_with_synonyms(all_cls_examples, synonyms, synonym_map)
            all_cls_examples += cls_synonym_examples

        return all_entity_examples + all_relation_examples + all_cls_examples

    def change_aspect_with_synonyms(self, examples, synonyms, synonym_map):
        synonym_examples = []
        for example in examples:
            prompt = example["prompt"]
            aspect, suffix = prompt.split("的", maxsplit=1)
            if aspect not in synonym_map.keys():
                continue
            synonym_cluster = synonyms[synonym_map[aspect]]
            for syn_aspect in synonym_cluster:
                if syn_aspect == aspect:
                    continue
                syn_prompt = syn_aspect + "的" +suffix
                syn_example = copy.deepcopy(example)
                syn_example["prompt"] = syn_prompt
                synonym_examples.append(syn_example)
        return synonym_examples
    

    def generate_cls_example(self,
                             text,
                             labels,
                             prompt_prefix,
                             options):
        random.shuffle(self.options)
        cls_options = ",".join(self.options)
        prompt = prompt_prefix + "[" + cls_options + "]"

        result_list = []
        example = {
            "content": text,
            "result_list": result_list,
            "prompt": prompt
        }

        for label in labels:
            start = prompt.rfind(label) - len(prompt) - 1
            end = start + len(label)
            result = {"text": label, "start": start, "end": end}
            example["result_list"].append(result)
        return example

    def add_full_negative_example(self,
                                  examples,
                                  texts,
                                  relation_prompt_list,
                                  predicate_set,
                                  subject_golden_list):
        with tqdm(total=len(relation_prompt_list)) as pbar:
            for i, relation_prompt in enumerate(relation_prompt_list):
                negative_sample = []
                for subject in subject_golden_list[i]:
                    for predicate in predicate_set:
                        # The relation prompt is constructed as follows:
                        # subject + "的" + predicate
                        prompt = subject + "的" + predicate
                        print(prompt)
                        if prompt not in relation_prompt:
                            negative_result = {
                                "content": texts[i],
                                "result_list": [],
                                "prompt": prompt
                            }
                            negative_sample.append(negative_result)
                examples[i].extend(negative_sample)
                pbar.update(1)
        return examples

    def add_entity_negative_example(self,
                                    examples,
                                    texts,
                                    prompts,
                                    label_set):
        negative_examples = []
        positive_examples = []
        with tqdm(total=len(prompts)) as pbar:
            for i, prompt in enumerate(prompts):
                redundants = list(set(label_set) ^ set(prompt))
                redundants.sort()

                ratio = self.negative_ratio
                if ratio > len(redundants):
                    ratio = len(redundants)
                idxs = random.sample(range(0, len(redundants)), ratio)

                for idx in idxs:
                    negative_result = {
                        "content": texts[i],
                        "result_list": [],
                        "prompt": redundants[idx]
                    }
                    negative_examples.append(negative_result)
                positive_examples.extend(examples[i])
                pbar.update(1)
        return positive_examples, negative_examples

    def add_relation_negative_example(self,
                                      redundants,
                                      text,
                                      ratio):
        added_example = []
        rest_example = []
        
        if ratio > len(redundants):
            ratio = len(redundants)
        
        all_idxs = [k for k in range(len(redundants))]
        idxs = random.sample(range(0, len(redundants)), ratio)
        rest_idxs = list(set(all_idxs) ^ set(idxs))

        for idx in idxs:
            negative_result = {
                "content": text,
                "result_list": [],
                "prompt": redundants[idx]
            }
            added_example.append(negative_result)

        for rest_idx in rest_idxs:
            negative_result = {
                "content": text,
                "result_list": [],
                "prompt": redundants[rest_idx]
            }
            rest_example.append(negative_result)

        return added_example, rest_example
    
    def add_cls_negative_example(self, texts, subject_name_set, subject_golden_list, with_full=False):
        negative_examples = []
        with tqdm(total=len(texts)) as pbar:
            for i, text in enumerate(texts):
                redundants = list(set(subject_name_set) ^ set(subject_golden_list[i]))
                redundants.sort()
                
                ratio = self.negative_ratio
                if not with_full:
                    if ratio > len(redundants):
                        ratio = len(redundants)
                    idxs = random.sample(range(0, len(redundants)), ratio)
                else:
                    idxs = [k for k in range(len(redundants))]

                for idx in idxs:
                    subject_name = redundants[idx]
                    prompt_prefix = subject_name + "的" +  self.prompt_prefix
                    negative_example = self.generate_cls_example(text, ["未提及"], prompt_prefix, self.options)
                    negative_examples.append(negative_example)
                pbar.update(1)
        return negative_examples


def do_convert():
    set_seed(args.seed)

    tic_time = time.time()
    if not os.path.exists(args.label_studio_file):
        raise ValueError("Please input the correct path of label studio file.")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if len(args.splits) != 0 and len(args.splits) != 3:
        raise ValueError("Only []/ len(splits)==3 accepted for splits.")

    def _check_sum(splits):
        return Decimal(str(splits[0])) + Decimal(str(splits[1])) + Decimal(
            str(splits[2])) == Decimal("1")

    if len(args.splits) == 3 and not _check_sum(args.splits):
        raise ValueError(
            "Please set correct splits, sum of elements in splits should be equal to 1."
        )

    with open(args.label_studio_file, "r", encoding="utf-8") as f:
        raw_examples = json.loads(f.read())

    if args.is_shuffle:
        indexes = np.random.permutation(len(raw_examples))
        index_list = indexes.tolist()
        raw_examples = [raw_examples[i] for i in indexes]

    i1, i2, _ = args.splits
    p1 = int(len(raw_examples) * i1)
    p2 = int(len(raw_examples) * (i1 + i2))

    train_ids = index_list[:p1]
    dev_ids = index_list[p1:p2]
    test_ids = index_list[p2:]

    with open(os.path.join(args.save_dir, "sample_index.json"), "w") as fp:
        maps = {
            "train_ids": train_ids,
            "dev_ids": dev_ids,
            "test_ids": test_ids
        }
        fp.write(json.dumps(maps))

    convertor = Convertor(negative_ratio=args.negative_ratio,
                          prompt_prefix=args.prompt_prefix,
                          options=args.options,
                          separator=args.separator)

    synonyms = None
    if args.synonym_file:
        synonyms = {}
        lines = load_txt(args.synonym_file)
        for line in lines:
            items = line.split()
            synonyms[items[0]] = items
    
    implicit_opinion_map = None
    if args.implicit_file:
        implicit_opinion_map = {}
        lines = load_txt(args.implicit_file)
        for line in lines:
            items = line.split()
            for item in items[1:]:
                implicit_opinion_map[item] = items[0]
    
    if args.task_type == "ext":
        train_examples = convertor.convert_ext_examples(raw_examples[:p1], synonyms=synonyms, implicit_opinion_map=implicit_opinion_map)
        dev_examples = convertor.convert_ext_examples(raw_examples[p1:p2], is_train=False, synonyms=synonyms, implicit_opinion_map=implicit_opinion_map)
        test_examples = convertor.convert_ext_examples(raw_examples[p2:], is_train=False, synonyms=synonyms, implicit_opinion_map=implicit_opinion_map)
        
    else:
        train_examples = convertor.convert_cls_examples(raw_examples[:p1])
        dev_examples = convertor.convert_cls_examples(raw_examples[p1:p2])
        test_examples = convertor.convert_cls_examples(raw_examples[p2:])
    
    def _save_examples(save_dir, file_name, examples):
        count = 0
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        logger.info("Save %d examples to %s." % (count, save_path))

    _save_examples(args.save_dir, "train.txt", train_examples)
    _save_examples(args.save_dir, "dev.txt", dev_examples)
    _save_examples(args.save_dir, "test.txt", test_examples)

    logger.info('Finished! It takes %.2f seconds' % (time.time() - tic_time))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_studio_file", default="./data/label_studio.json", type=str, help="The annotation file exported from label studio platform.")
    parser.add_argument("--synonym_file", default="./data/synonyms.txt", type=str, help="The synonmy file of aspect to support aspect aggregation.")
    parser.add_argument("--implicit_file", default="./data/implicit_opinions.txt", type=str, help="The implicit opinion file whose aspect not be mentioned in text, to support extraction of implicit opinion.")
    parser.add_argument("--save_dir", default="./data", type=str, help="The path of data that you wanna save.")
    parser.add_argument("--negative_ratio", default=5, type=int, help="Used only for the extraction task, the ratio of positive and negative samples, number of negtive samples = negative_ratio * number of positive samples")
    parser.add_argument("--splits", default=[0.8, 0.1, 0.1], type=float, nargs="*", help="The ratio of samples in datasets. [0.6, 0.2, 0.2] means 60% samples used for training, 20% for evaluation and 20% for test.")
    parser.add_argument("--task_type", choices=['ext', 'cls'], default="ext", type=str, help="Select task type, ext for the extraction task and cls for the classification task, defaults to ext.")
    parser.add_argument("--options", default=["正向", "负向", "未提及"], type=str, nargs="+", help="Used only for the classification task, the options for classification")
    parser.add_argument("--prompt_prefix", default="情感倾向", type=str, help="Used only for the classification task, the prompt prefix for classification")
    parser.add_argument("--is_shuffle", default=True, type=bool, help="Whether to shuffle the labeled dataset, defaults to True.")
    parser.add_argument("--layout_analysis", default=False, type=bool, help="Enable layout analysis to optimize the order of OCR result.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization")
    parser.add_argument("--separator", type=str, default='##', help="Used only for entity/aspect-level classification task, separator for entity label and classification label")

    args = parser.parse_args()
    # yapf: enable

    do_convert()