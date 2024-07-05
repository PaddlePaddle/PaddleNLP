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

import copy
import json
import os
import random

import numpy as np
import paddle
from data_collator import DataCollator

from paddlenlp.taskflow.utils import SchemaTree
from paddlenlp.utils.log import logger

criteria_map = {
    "entity_extraction": "entity_f1",
    "opinion_extraction": "relation_f1",  # (Aspect, Sentiment, Opinion)
    "relation_extraction": "relation_f1",  # (Subject, Predicate, Object)
    "event_extraction": "relation_f1",  # (Trigger, Role, Argument)
}


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def reader(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            yield json_line


def save_model_config(save_dir, model_config):
    model_config_file = os.path.join(save_dir, "model_config.json")
    with open(model_config_file, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(model_config, ensure_ascii=False, indent=2))


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def get_label_maps(task_type="relation_extraction", label_maps_path=None):
    with open(label_maps_path, "r", encoding="utf-8") as fp:
        label_maps = json.load(fp)
    if task_type == "entity_extraction":
        entity2id = label_maps["entity2id"]
        id2entity = {idx: t for t, idx in entity2id.items()}
        label_maps["id2entity"] = id2entity
    else:
        entity2id = label_maps["entity2id"]
        relation2id = (
            label_maps["relation2id"]
            if task_type in ["relation_extraction", "event_extraction"]
            else label_maps["sentiment2id"]
        )
        id2entity = {idx: t for t, idx in entity2id.items()}
        id2relation = {idx: t for t, idx in relation2id.items()}
        label_maps["id2entity"] = id2entity
        label_maps["id2relation"] = id2relation
    return label_maps


def create_dataloader(
    dataset, tokenizer, max_seq_len=128, batch_size=1, label_maps=None, mode="train", task_type="relation_extraction"
):
    def tokenize_and_align_train_labels(example):
        tokenized_inputs = tokenizer(
            example["text"],
            max_length=max_seq_len,
            padding=False,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_offsets_mapping=True,
        )
        offset_mapping = tokenized_inputs["offset_mapping"]

        ent_labels = []
        for e in example["entity_list"]:
            _start, _end = e["start_index"], e["start_index"] + len(e["text"]) - 1
            start = map_offset(_start, offset_mapping)
            end = map_offset(_end, offset_mapping)
            if start == -1 or end == -1:
                continue
            label = label_maps["entity2id"][e["type"]]
            ent_labels.append([label, start, end])

        outputs = {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": {"ent_labels": ent_labels, "rel_labels": []},
        }

        if task_type in ["relation_extraction", "event_extraction"]:
            rel_labels = []
            for r in example["spo_list"]:
                _sh, _oh = r["subject_start_index"], r["object_start_index"]
                _st, _ot = _sh + len(r["subject"]) - 1, _oh + len(r["object"]) - 1
                sh = map_offset(_sh, offset_mapping)
                st = map_offset(_st, offset_mapping)
                oh = map_offset(_oh, offset_mapping)
                ot = map_offset(_ot, offset_mapping)
                if sh == -1 or st == -1 or oh == -1 or ot == -1:
                    continue
                p = label_maps["relation2id"][r["predicate"]]
                rel_labels.append([sh, st, p, oh, ot])
            outputs["labels"]["rel_labels"] = rel_labels
        elif task_type == "opinion_extraction":
            rel_labels = []
            for r in example["aso_list"]:
                _ah, _oh = r["aspect_start_index"], r["opinion_start_index"]
                _at, _ot = _ah + len(r["aspect"]) - 1, _oh + len(r["opinion"]) - 1
                ah = map_offset(_ah, offset_mapping)
                at = map_offset(_at, offset_mapping)
                oh = map_offset(_oh, offset_mapping)
                ot = map_offset(_ot, offset_mapping)
                if ah == -1 or at == -1 or oh == -1 or ot == -1:
                    continue

                s = label_maps["sentiment2id"][r["sentiment"]]
                rel_labels.append([ah, at, s, oh, ot])
            outputs["labels"]["rel_labels"] = rel_labels
        return outputs

    def tokenize(example):
        tokenized_inputs = tokenizer(
            example["text"],
            max_length=max_seq_len,
            padding=False,
            truncation=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )
        tokenized_inputs["text"] = example["text"]
        return tokenized_inputs

    if mode == "train":
        dataset = dataset.map(tokenize_and_align_train_labels)
    else:
        dataset_copy = copy.deepcopy(dataset)
        dataset = dataset.map(tokenize)

    data_collator = DataCollator(tokenizer, label_maps=label_maps, task_type=task_type)

    shuffle = True if mode == "train" else False
    batch_sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(
        dataset=dataset, batch_sampler=batch_sampler, collate_fn=data_collator, num_workers=0, return_list=True
    )
    if mode != "train":
        dataloader.dataset.raw_data = dataset_copy
    return dataloader


def postprocess(batch_outputs, offset_mappings, texts, label_maps, task_type="relation_extraction"):
    if task_type == "entity_extraction":
        batch_ent_results = []
        for entity_output, offset_mapping, text in zip(batch_outputs[0].numpy(), offset_mappings, texts):
            entity_output[:, [0, -1]] -= np.inf
            entity_output[:, :, [0, -1]] -= np.inf
            ent_list = []
            for l, start, end in zip(*np.where(entity_output > 0.0)):
                start, end = (offset_mapping[start][0], offset_mapping[end][-1])
                ent = {"text": text[start:end], "type": label_maps["id2entity"][l], "start_index": start}
                ent_list.append(ent)
            batch_ent_results.append(ent_list)
        return batch_ent_results
    else:
        batch_ent_results = []
        batch_rel_results = []
        for entity_output, head_output, tail_output, offset_mapping, text in zip(
            batch_outputs[0].numpy(),
            batch_outputs[1].numpy(),
            batch_outputs[2].numpy(),
            offset_mappings,
            texts,
        ):
            entity_output[:, [0, -1]] -= np.inf
            entity_output[:, :, [0, -1]] -= np.inf
            ents = set()
            ent_list = []
            for l, start, end in zip(*np.where(entity_output > 0.0)):
                ents.add((start, end))
                start, end = (offset_mapping[start][0], offset_mapping[end][-1])
                ent = {"text": text[start:end], "type": label_maps["id2entity"][l], "start_index": start}
                ent_list.append(ent)
            batch_ent_results.append(ent_list)

            rel_list = []
            for sh, st in ents:
                for oh, ot in ents:
                    p1s = np.where(head_output[:, sh, oh] > 0.0)[0]
                    p2s = np.where(tail_output[:, st, ot] > 0.0)[0]
                    ps = set(p1s) & set(p2s)
                    for p in ps:
                        if task_type in ["relation_extraction", "event_extraction"]:
                            rel = {
                                "subject": text[offset_mapping[sh][0] : offset_mapping[st][1]],
                                "predicate": label_maps["id2relation"][p],
                                "object": text[offset_mapping[oh][0] : offset_mapping[ot][1]],
                                "subject_start_index": offset_mapping[sh][0],
                                "object_start_index": offset_mapping[oh][0],
                            }
                        else:
                            rel = {
                                "aspect": text[offset_mapping[sh][0] : offset_mapping[st][1]],
                                "sentiment": label_maps["id2relation"][p],
                                "opinion": text[offset_mapping[oh][0] : offset_mapping[ot][1]],
                                "aspect_start_index": offset_mapping[sh][0],
                                "opinion_start_index": offset_mapping[oh][0],
                            }
                        rel_list.append(rel)
            batch_rel_results.append(rel_list)
        return (batch_ent_results, batch_rel_results)


def build_tree(schema, name="root"):
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
                        "but {} received".format(type(v))
                    )
                schema_tree.add_child(build_tree(child, name=k))
        else:
            raise TypeError("Invalid schema, element should be string or dict, " "but {} received".format(type(s)))
    return schema_tree


def schema2label_maps(task_type, schema=None):
    if schema and isinstance(schema, dict):
        schema = [schema]

    label_maps = {}
    if task_type == "entity_extraction":
        entity2id = {}
        for s in schema:
            entity2id[s] = len(entity2id)

        label_maps["entity2id"] = entity2id
    elif task_type == "opinion_extraction":
        schema = ["观点词", {"评价维度": ["观点词", "情感倾向[正向,负向]"]}]
        logger.info("Opinion extraction does not support custom schema, the schema is default to %s." % schema)
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

        entity2id["object"] = len(entity2id)
        label_maps["entity2id"] = entity2id
        label_maps["relation2id"] = relation2id

    label_maps["schema"] = schema
    return label_maps


def anno2distill(json_lines, task_type, label_maps=None, platform="label_studio"):
    if platform == "label_studio":
        return label_studio2distill(json_lines, task_type, label_maps)
    else:
        return doccano2distill(json_lines, task_type, label_maps)


def label_studio2distill(json_lines, task_type, label_maps=None):
    """Convert label-studio to distill format"""
    if task_type == "opinion_extraction":
        outputs = []
        for json_line in json_lines:
            id2ent = {}
            text = json_line["data"]["text"]
            output = {"text": text}
            entity_list = []
            aso_list = []
            annos = json_line["annotations"][0]["result"]
            for anno in annos:
                if anno["type"] == "labels":
                    ent_text = text[anno["value"]["start"] : anno["value"]["end"]]
                    ent_type_gather = anno["value"]["labels"][0].split("##")
                    if len(ent_type_gather) == 2:
                        ent_type, ent_senti = ent_type_gather
                    else:
                        ent_type = ent_type_gather[0]
                        ent_senti = None
                    ent = {"text": ent_text, "type": ent_type, "start_index": anno["value"]["start"]}
                    id2ent[anno["id"]] = ent
                    id2ent[anno["id"]]["sentiment"] = ent_senti
                    entity_list.append(ent)
                else:
                    _aspect = id2ent[anno["from_id"]]
                    if _aspect["sentiment"]:
                        _opinion = id2ent[anno["to_id"]]
                        rel = {
                            "aspect": _aspect["text"],
                            "sentiment": _aspect["sentiment"],
                            "opinion": _opinion["text"],
                            "aspect_start_index": _aspect["start_index"],
                            "opinion_start_index": _opinion["start_index"],
                        }
                        aso_list.append(rel)
                    output["aso_list"] = aso_list
            output["entity_list"] = entity_list
            output["aso_list"] = aso_list
            outputs.append(output)
    else:
        outputs = []
        for json_line in json_lines:
            id2ent = {}
            text = json_line["data"]["text"]
            output = {"text": text}
            entity_list = []
            spo_list = []
            annos = json_line["annotations"][0]["result"]
            for anno in annos:
                if anno["type"] == "labels":
                    ent_text = text[anno["value"]["start"] : anno["value"]["end"]]
                    ent_label = anno["value"]["labels"][0]
                    ent_type = "object" if ent_label not in label_maps["entity2id"].keys() else ent_label
                    ent = {"text": ent_text, "type": ent_type, "start_index": anno["value"]["start"]}
                    id2ent[anno["id"]] = ent
                    entity_list.append(ent)
                else:
                    _subject = id2ent[anno["from_id"]]
                    _object = id2ent[anno["to_id"]]
                    rel = {
                        "subject": _subject["text"],
                        "predicate": anno["labels"][0],
                        "object": _object["text"],
                        "subject_start_index": _subject["start_index"],
                        "object_start_index": _object["start_index"],
                    }
                    spo_list.append(rel)
            output["entity_list"] = entity_list
            output["spo_list"] = spo_list
            outputs.append(output)
    return outputs


def doccano2distill(json_lines, task_type, label_maps=None):
    """Convert doccano to distill format"""
    if task_type == "opinion_extraction":
        outputs = []
        for json_line in json_lines:
            id2ent = {}
            text = json_line["text"]
            output = {"text": text}
            entity_list = []
            entities = json_line["entities"]
            for entity in entities:
                ent_text = text[entity["start_offset"] : entity["end_offset"]]
                ent_type_gather = entity["label"].split("##")
                if len(ent_type_gather) == 2:
                    ent_type, ent_senti = ent_type_gather
                else:
                    ent_type = ent_type_gather[0]
                    ent_senti = None
                ent = {"text": ent_text, "type": ent_type, "start_index": entity["start_offset"]}
                id2ent[entity["id"]] = ent
                id2ent[entity["id"]]["sentiment"] = ent_senti
                entity_list.append(ent)
            output["entity_list"] = entity_list
            aso_list = []
            relations = json_line["relations"]
            for relation in relations:
                _aspect = id2ent[relation["from_id"]]
                if _aspect["sentiment"]:
                    _opinion = id2ent[relation["to_id"]]
                    rel = {
                        "aspect": _aspect["text"],
                        "sentiment": _aspect["sentiment"],
                        "opinion": _opinion["text"],
                        "aspect_start_index": _aspect["start_index"],
                        "opinion_start_index": _opinion["start_index"],
                    }
                    aso_list.append(rel)
            output["aso_list"] = aso_list
            outputs.append(output)
    else:
        outputs = []
        for json_line in json_lines:
            id2ent = {}
            text = json_line["text"]
            output = {"text": text}
            entity_list = []
            entities = json_line["entities"]
            for entity in entities:
                ent_text = text[entity["start_offset"] : entity["end_offset"]]
                if entity["label"] not in label_maps["entity2id"].keys():
                    if task_type == "entity_extraction":
                        logger.warning(
                            "Found undefined label type. The setting of schema should contain all the label types in annotation file export from annotation platform."
                        )
                        continue
                    else:
                        ent_type = "object"
                else:
                    ent_type = entity["label"]
                ent = {"text": ent_text, "type": ent_type, "start_index": entity["start_offset"]}
                id2ent[entity["id"]] = ent
                entity_list.append(ent)
            output["entity_list"] = entity_list
            spo_list = []
            relations = json_line["relations"]
            for relation in relations:
                _subject = id2ent[relation["from_id"]]
                _object = id2ent[relation["to_id"]]
                rel = {
                    "subject": _subject["text"],
                    "predicate": relation["type"],
                    "object": _object["text"],
                    "subject_start_index": _subject["start_index"],
                    "object_start_index": _object["start_index"],
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
            pred = line
            output = {"text": texts[i]}

            entity_list = []
            aso_list = []
            for key1 in pred.keys():
                for s in pred[key1]:
                    ent = {"text": s["text"], "type": key1, "start_index": s["start"]}
                    entity_list.append(ent)

                    if (
                        "relations" in s.keys()
                        and "观点词" in s["relations"].keys()
                        and "情感倾向[正向,负向]" in s["relations"].keys()
                    ):
                        for o in s["relations"]["观点词"]:
                            rel = {
                                "aspect": s["text"],
                                "sentiment": s["relations"]["情感倾向[正向,负向]"][0]["text"],
                                "opinion": o["text"],
                                "aspect_start_index": s["start"],
                                "opinion_start_index": o["start"],
                            }
                            aso_list.append(rel)

                            ent = {"text": o["text"], "type": "观点词", "start_index": o["start"]}
                            entity_list.append(ent)
            output["entity_list"] = entity_list
            output["aso_list"] = aso_list
            outputs.append(output)
    else:
        outputs = []
        for i, line in enumerate(infer_results):
            pred = line
            output = {"text": texts[i]}

            entity_list = []
            spo_list = []
            for key1 in pred.keys():
                for s in pred[key1]:
                    ent = {"text": s["text"], "type": key1, "start_index": s["start"]}
                    entity_list.append(ent)
                    if "relations" in s.keys():
                        for key2 in s["relations"].keys():
                            for o1 in s["relations"][key2]:
                                if "start" in o1.keys():
                                    rel = {
                                        "subject": s["text"],
                                        "predicate": key2,
                                        "object": o1["text"],
                                        "subject_start_index": s["start"],
                                        "object_start_index": o1["start"],
                                    }
                                    spo_list.append(rel)

                                    if "relations" not in o1.keys():
                                        ent = {"text": o1["text"], "type": "object", "start_index": o1["start"]}
                                        entity_list.append(ent)
                                    else:
                                        ent = {"text": o1["text"], "type": key2, "start_index": o1["start"]}
                                        entity_list.append(ent)
                                        for key3 in o1["relations"].keys():
                                            for o2 in o1["relations"][key3]:
                                                ent = {
                                                    "text": o2["text"],
                                                    "type": "object",
                                                    "start_index": o2["start"],
                                                }
                                                entity_list.append(ent)

                                                rel = {
                                                    "subject": o1["text"],
                                                    "predicate": key3,
                                                    "object": o2["text"],
                                                    "subject_start_index": o1["start"],
                                                    "object_start_index": o2["start"],
                                                }
                                                spo_list.append(rel)
            output["entity_list"] = entity_list
            output["spo_list"] = spo_list
            outputs.append(output)
    return outputs
