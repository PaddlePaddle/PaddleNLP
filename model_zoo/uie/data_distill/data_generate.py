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
import json
import math
import random
import argparse
from tqdm import tqdm

import numpy as np
import paddle
from paddlenlp import Taskflow
from paddlenlp.taskflow.utils import SchemaTree
from paddlenlp.utils.log import logger


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_tree(schema, name='root'):
    """
    Build the schema tree.
    """
    schema_tree = SchemaTree(name)
    for s in schema:
        if isinstance(s, str):
            schema_tree.add_child(SchemaTree(s))
        elif isinstance(s, dict):
            for k, v in s.items():
                if isinstance(v, str):
                    child = [v]
                elif isinstance(v, list):
                    child = v
                else:
                    raise TypeError(
                        "Invalid schema, value for each key:value pairs should be list or string"
                        "but {} received".format(type(v)))
                schema_tree.add_child(build_tree(child, name=k))
        else:
            raise TypeError("Invalid schema, element should be string or dict, "
                            "but {} received".format(type(s)))
    return schema_tree


def schema2label_maps(task_type, schema=None):
    label_maps = {}
    if task_type == "entity_extraction":
        entity2id = {}
        for s in schema:
            entity2id[s] = len(entity2id)

        label_maps["entity2id"] = entity2id
    elif task_type == "opinion_extraction":
        label_maps["entity2id"] = {"评价维度": 0, "观点词": 1}
        label_maps["sentiment2id"] = {"正向": 0, "负向": 1}
    else:
        entity2id = {}
        relation2id = {}
        schema_tree = build_tree(schema)
        schema_list = schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)

            if node.name not in entity2id.keys() and len(node.children) != 0:
                entity2id[node.name] = len(entity2id)

            for child in node.children:
                if child.name not in relation2id.keys():
                    relation2id[child.name] = len(relation2id)
                schema_list.append(child)

        entity2id['OBJECT'] = len(entity2id)
        label_maps["entity2id"] = entity2id
        label_maps["relation2id"] = relation2id

    label_maps["schema"] = schema
    return label_maps


def doccano2distill(json_lines, task_type, label_maps=None):
    """Convert doccano to distill format"""
    if task_type == "opinion_extraction":
        outputs = []
        for json_line in json_lines:
            id2ent = {}
            text = json_line['text']
            output = {"text": text}

            entity_list = []
            entities = json_line['entities']
            for entity in entities:
                ent_text = text[entity['start_offset']:entity['end_offset']]

                ent_type_gather = entity['label'].split("##")
                if len(ent_type_gather) == 2:
                    ent_type, ent_senti = ent_type_gather
                else:
                    ent_type = ent_type_gather[0]
                    ent_senti = None

                ent_start_idx = entity['start_offset']

                id2ent[entity['id']] = {
                    "text": ent_text,
                    "type": ent_type,
                    "start_index": ent_start_idx,
                    "sentiment": ent_senti
                }

                ent = {
                    "text": ent_text,
                    "type": ent_type,
                    "start_index": ent_start_idx
                }

                entity_list.append(ent)
            output["entity_list"] = entity_list

            aso_list = []
            relations = json_line['relations']
            for relation in relations:
                _aspect = id2ent[relation["from_id"]]
                if _aspect['sentiment']:
                    _opinion = id2ent[relation["to_id"]]
                    rel = {
                        "aspect": _aspect['text'],
                        "sentiment": _aspect['sentiment'],
                        "opinion": _opinion['text'],
                        "aspect_start_index": _aspect["start_index"],
                        "opinion_start_index": _opinion["start_index"]
                    }
                    aso_list.append(rel)
            output["aso_list"] = aso_list
            outputs.append(output)
    else:
        outputs = []
        for json_line in json_lines:
            id2ent = {}
            text = json_line['text']
            output = {"text": text}

            entity_list = []
            entities = json_line['entities']
            for entity in entities:
                ent_text = text[entity['start_offset']:entity['end_offset']]
                ent_type = "OBJECT" if entity['label'] not in label_maps[
                    'entity2id'].keys() else entity['label']
                ent_start_idx = entity['start_offset']

                id2ent[entity['id']] = {
                    "text": ent_text,
                    "type": ent_type,
                    "start_index": ent_start_idx
                }

                ent = {
                    "text": ent_text,
                    "type": ent_type,
                    "start_index": ent_start_idx
                }

                entity_list.append(ent)
            output["entity_list"] = entity_list

            spo_list = []
            relations = json_line['relations']
            for relation in relations:
                _subject = id2ent[relation["from_id"]]
                _object = id2ent[relation["to_id"]]
                rel = {
                    "subject": _subject['text'],
                    "predicate": relation['type'],
                    "object": _object['text'],
                    "subject_start_index": _subject["start_index"],
                    "object_start_index": _object["start_index"]
                }
                spo_list.append(rel)
            output["spo_list"] = spo_list
            outputs.append(output)
    return outputs


def synthetic2distill(texts, infer_results, task_type, label_maps=None):
    """Convert synthetic data to distill format"""
    if task_type == "opinion_extraction":
        outputs = []
        for i, line in enumerate(infer_results):
            pred = line[0]
            output = {"text": texts[i]}

            entity_list = []
            aso_list = []
            for key1 in pred.keys():
                for s in pred[key1]:
                    ent = {
                        "text": s["text"],
                        "type": key1,
                        "start_index": s["start"]
                    }
                    entity_list.append(ent)

                    if "relations" in s.keys() and "观点词" in s["relations"].keys(
                    ) and "情感倾向[正向,负向]" in s["relations"].keys():
                        for o in s["relations"]["观点词"]:
                            rel = {
                                "aspect":
                                s["text"],
                                "sentiment":
                                s["relations"]["情感倾向[正向,负向]"][0]["text"],
                                "opinion":
                                o["text"],
                                "aspect_start_index":
                                s["start"],
                                "opinion_start_index":
                                o["start"]
                            }
                            aso_list.append(rel)

                            ent = {
                                "text": o["text"],
                                "type": "观点词",
                                "start_index": o["start"]
                            }
                            entity_list.append(ent)
            output["entity_list"] = entity_list
            output["aso_list"] = aso_list
            outputs.append(output)
    else:
        outputs = []
        for i, line in enumerate(infer_results):
            pred = line[0]
            output = {"text": texts[i]}

            entity_list = []
            spo_list = []
            for key1 in pred.keys():
                for s in pred[key1]:
                    ent = {
                        "text": s['text'],
                        "type": key1,
                        "start_index": s['start']
                    }
                    entity_list.append(ent)
                    if "relations" in s.keys():
                        for key2 in s['relations'].keys():
                            for o1 in s['relations'][key2]:
                                rel = {
                                    "subject": s['text'],
                                    "predicate": key2,
                                    "object": o1['text'],
                                    "subject_start_index": s['start'],
                                    "object_start_index": o1['start']
                                }
                                spo_list.append(rel)

                                if 'relations' not in o1.keys():
                                    ent = {
                                        "text": o1['text'],
                                        "type": "OBJECT",
                                        "start_index": o1['start']
                                    }
                                    entity_list.append(ent)
                                else:
                                    ent = {
                                        "text": o1['text'],
                                        "type": key2,
                                        "start_index": o1['start']
                                    }
                                    entity_list.append(ent)
                                    for key3 in o1['relations'].keys():
                                        for o2 in o1['relations'][key3]:
                                            ent = {
                                                "text": o2['text'],
                                                "type": "OBJECT",
                                                "start_index": o2['start']
                                            }
                                            entity_list.append(ent)

                                            rel = {
                                                "subject": o1['text'],
                                                "predicate": key3,
                                                "object": o2['text'],
                                                "subject_start_index":
                                                o1['start'],
                                                "object_start_index":
                                                o2['start']
                                            }
                                            spo_list.append(rel)
            output["entity_list"] = entity_list
            output["spo_list"] = spo_list
            outputs.append(output)
    return outputs


def do_generate():
    set_seed(args.seed)

    # Generate closed-domain label maps
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    label_maps = schema2label_maps(args.task_type, schema=args.schema)
    label_maps_path = os.path.join(args.output_dir, "label_maps.json")

    # Save closed-domain label maps file
    with open(label_maps_path, "w") as fp:
        fp.write(json.dumps(label_maps, ensure_ascii=False))

    # Load doccano file and convert to distill format
    sample_index = json.loads(
        open(os.path.join(args.data_dir, "sample_index.json")).readline())

    train_ids = sample_index["train_ids"]
    dev_ids = sample_index["dev_ids"]
    test_ids = sample_index["test_ids"]

    json_lines = []
    with open(os.path.join(args.data_dir, "doccano_ext.json")) as fp:
        for line in fp:
            json_lines.append(json.loads(line))

    train_lines = [json_lines[i] for i in train_ids]
    train_lines = doccano2distill(train_lines, args.task_type, label_maps)

    dev_lines = [json_lines[i] for i in dev_ids]
    dev_lines = doccano2distill(dev_lines, args.task_type, label_maps)

    test_lines = [json_lines[i] for i in test_ids]
    test_lines = doccano2distill(test_lines, args.task_type, label_maps)

    # Load trained UIE model
    uie = Taskflow("information_extraction",
                   schema=args.schema,
                   task_path=args.model_path)

    # Generate synthetic data
    texts = open(os.path.join(args.data_dir, "unlabeled_data.txt")).readlines()

    actual_ratio = math.ceil(len(texts) / len(train_lines))
    if actual_ratio <= args.synthetic_ratio or args.synthetic_ratio == -1:
        infer_texts = texts
    else:
        idxs = random.sample(range(0, len(texts)),
                             args.synthetic_ratio * len(train_lines))
        infer_texts = [texts[i] for i in idxs]

    infer_results = []
    for text in tqdm(infer_texts, desc="Predicting: ", leave=False):
        infer_results.append(uie(text))

    train_synthetic_lines = synthetic2distill(texts, infer_results,
                                              args.task_type)

    def _save_examples(save_dir, file_name, examples):
        count = 0
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        logger.info("Save %d examples to %s." % (count, save_path))

    # Concat origin and synthetic data
    train_lines.extend(train_synthetic_lines)

    _save_examples(args.output_dir, "train_data.json", train_lines)
    _save_examples(args.output_dir, "dev_data.json", dev_lines)
    _save_examples(args.output_dir, "test_data.json", test_lines)


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data", type=str, help="")
    parser.add_argument("--model_path", type=str, default="../checkpoint/model_best", help="The path of saved model that you want to load.")
    parser.add_argument("--output_dir", default="./distill_task", type=str, help="The path of data that you wanna save.")
    parser.add_argument("--synthetic_ratio", default=10, type=int, help="The ratio of labeled and synthetic samples.")
    parser.add_argument("--task_type", choices=['relation_extraction', 'event_extraction', 'entity_extraction', 'opinion_extraction'], default="entity_extraction", type=str, help="Select the training task type.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization")

    args = parser.parse_args()
    # yapf: enable

    # Define your schema here
    schema = ["观点词", {"评价维度": ["观点词", "情感倾向[正向,负向]"]}]

    args.schema = schema

    do_generate()
