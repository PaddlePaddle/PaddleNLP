# coding=utf-8
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

import base64
import json
import os
import random

import numpy as np
import paddle
from data_collator import DataCollator

from paddlenlp.taskflow.utils import SchemaTree
from paddlenlp.utils.ie_utils import map_offset, pad_image_data


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class PreProcessor:
    def __init__(self):
        pass

    @classmethod
    def preprocess_doc(
        self,
        examples,
        tokenizer=None,
        max_length=512,
        doc_stride=256,
        label_maps=None,
        lang="ch",
        with_label=True,
    ):
        def _process_bbox(tokens, bbox_lines, offset_mapping, offset_bias):
            bbox_list = [[0, 0, 0, 0] for x in range(len(tokens))]

            for index, bbox in enumerate(bbox_lines):
                index_token = map_offset(index + offset_bias, offset_mapping)
                if 0 <= index_token < len(bbox_list):
                    bbox_list[index_token] = bbox
            return bbox_list

        tokenized_examples = []
        for example in examples:
            content = example["text"]
            bbox_lines = example["bbox"]
            image_buff_string = example["image"]
            doc_id = example["id"]

            image_data = base64.b64decode(image_buff_string.encode("utf8"))
            padded_image = pad_image_data(image_data)

            all_doc_tokens = tokenizer.tokenize(content)
            offset_mapping = tokenizer.get_offset_mapping(content)

            all_doc_token_boxes = _process_bbox(all_doc_tokens, bbox_lines, offset_mapping, 0)

            start_offset = 0
            doc_spans = []
            max_tokens_for_doc = max_length - 2
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append({"start": start_offset, "length": length})
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride, max_tokens_for_doc)

            for doc_span in doc_spans:
                tokens = [tokenizer.cls_token]
                token_boxes = [[0, 0, 0, 0]]
                doc_start = doc_span["start"]
                doc_end = doc_span["start"] + doc_span["length"] - 1

                for i in range(doc_span["length"]):
                    token_index = doc_span["start"] + i
                    tokens.append(all_doc_tokens[token_index])
                    token_boxes.append(all_doc_token_boxes[token_index])

                tokens.append(tokenizer.sep_token)
                token_boxes.append([0, 0, 0, 0])

                input_mask = [1] * len(tokens)

                while len(tokens) < max_length:
                    tokens.append(tokenizer.pad_token)
                    input_mask.append(0)
                    token_boxes.append([0, 0, 0, 0])

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                assert len(input_ids) == max_length
                assert len(input_mask) == max_length
                assert len(token_boxes) == max_length

                input_list = {
                    "id": doc_id,
                    "input_ids": input_ids,
                    "attention_mask": input_mask,
                    "bbox": token_boxes,
                    "image": padded_image,
                }

                if with_label:
                    ent_labels = []
                    for e in example["entity_list"]:
                        _start, _end = e["start_index"], e["start_index"] + len(e["text"]) - 1
                        start = map_offset(_start, offset_mapping)
                        end = map_offset(_end, offset_mapping)
                        if not (start >= doc_start and end <= doc_end):
                            continue
                        if start == -1 or end == -1:
                            continue
                        label = label_maps["entity2id"][e["type"]]
                        ent_labels.append([label, start - doc_start + 1, end - doc_start + 1])

                    rel_labels = []
                    for r in example["spo_list"]:
                        _sh, _oh = r["subject_start_index"], r["object_start_index"]
                        _st, _ot = _sh + len(r["subject"]) - 1, _oh + len(r["object"]) - 1
                        sh = map_offset(_sh, offset_mapping)
                        st = map_offset(_st, offset_mapping)
                        oh = map_offset(_oh, offset_mapping)
                        ot = map_offset(_ot, offset_mapping)
                        if not (sh >= doc_start and st <= doc_end) or not (oh >= doc_start and ot <= doc_end):
                            continue
                        if sh == -1 or st == -1 or oh == -1 or ot == -1:
                            continue
                        p = label_maps["relation2id"][r["predicate"]]
                        rel_labels.append(
                            [
                                sh - doc_start + 1,
                                st - doc_start + 1,
                                p,
                                oh - doc_start + 1,
                                ot - doc_start + 1,
                            ]
                        )

                    input_list["labels"] = {"ent_labels": ent_labels, "rel_labels": rel_labels}

                tokenized_examples.append(input_list)
        return tokenized_examples


def reader(data_path, tokenizer, max_length=512, doc_stride=128, label_maps=None, lang="ch"):
    with open(data_path, "r", encoding="utf-8") as f:
        examples = []
        for line in f:
            example = json.loads(line)
            examples.append(example)

    tokenized_examples = PreProcessor.preprocess_doc(
        examples,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
        label_maps=label_maps,
        lang=lang,
    )
    for tokenized_example in tokenized_examples:
        yield tokenized_example


def save_model_config(save_dir, model_config):
    model_config_file = os.path.join(save_dir, "model_config.json")
    with open(model_config_file, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(model_config, ensure_ascii=False, indent=2))


def get_label_maps(label_maps_path=None):
    with open(label_maps_path, "r", encoding="utf-8") as fp:
        label_maps = json.load(fp)
    entity2id = label_maps["entity2id"]
    relation2id = label_maps["relation2id"]
    id2entity = {idx: t for t, idx in entity2id.items()}
    id2relation = {idx: t for t, idx in relation2id.items()}
    label_maps["id2entity"] = id2entity
    label_maps["id2relation"] = id2relation
    return label_maps


def create_dataloader(dataset, tokenizer=None, label_maps=None, batch_size=1, mode="train"):
    shuffle = True if mode == "train" else False
    batch_sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    data_collator = DataCollator(tokenizer, label_maps=label_maps)

    dataloader = paddle.io.DataLoader(
        dataset=dataset, batch_sampler=batch_sampler, collate_fn=data_collator, num_workers=0, return_list=True
    )
    return dataloader


def postprocess(batch_outputs, offset_mappings, texts, label_maps):
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
                    rel = {
                        "subject": text[offset_mapping[sh][0] : offset_mapping[st][1]],
                        "predicate": label_maps["id2relation"][p],
                        "object": text[offset_mapping[oh][0] : offset_mapping[ot][1]],
                        "subject_start_index": offset_mapping[sh][0],
                        "object_start_index": offset_mapping[oh][0],
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


def schema2label_maps(schema=None):
    if schema and isinstance(schema, dict):
        schema = [schema]

    label_maps = {}

    entity2id = {}
    relation2id = {}
    schema_tree = build_tree(schema)
    schema_list = schema_tree.children[:]

    while len(schema_list) > 0:
        node = schema_list.pop(0)

        if not node.parent_relations:
            entity2id[node.name] = len(entity2id)
            parent_relations = node.name
        elif node.name not in entity2id.keys() and len(node.children) != 0:
            entity2id[node.name] = len(entity2id)

        for child in node.children:
            child.parent_relations = parent_relations
            if child.name not in relation2id.keys():
                relation2id[child.name] = len(relation2id)
            schema_list.append(child)

    if relation2id:
        entity2id["object"] = len(entity2id)
    label_maps["entity2id"] = entity2id
    label_maps["relation2id"] = relation2id

    label_maps["schema"] = schema
    return label_maps
