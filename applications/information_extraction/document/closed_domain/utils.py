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
import paddle.nn.functional as F
from paddlenlp.taskflow.utils import SchemaTree
from paddlenlp.utils.ie_utils import map_offset, pad_image_data

from data_collator import DataCollator


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class Processor:
    def __init__(self):
        pass

    @classmethod
    def preprocess_doc(
        self,
        examples,
        tokenizer=None,
        max_seq_len=512,
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

            all_doc_tokens = []
            all_offset_mapping = []
            last_offset = 0
            for char_index, (char, bbox) in enumerate(zip(content, bbox_lines)):
                if char_index == 0:
                    prev_bbox = bbox
                    this_text_line = char
                    continue

                if all(bbox[x] == prev_bbox[x] for x in range(4)):
                    this_text_line += char
                else:
                    cur_offset_mapping = tokenizer.get_offset_mapping(this_text_line)
                    for i, sub_list in enumerate(cur_offset_mapping):
                        if i == 0:
                            org_offset = sub_list[1]
                        else:
                            if sub_list[0] != org_offset:
                                last_offset += 1
                            org_offset = sub_list[1]
                        all_offset_mapping += [[last_offset, sub_list[1] - sub_list[0] + last_offset]]
                        last_offset = all_offset_mapping[-1][-1]
                    all_doc_tokens += tokenizer.tokenize(this_text_line)
                    this_text_line = char
                prev_bbox = bbox
            if len(this_text_line) > 0:
                cur_offset_mapping = tokenizer.get_offset_mapping(this_text_line)
                for i, sub_list in enumerate(cur_offset_mapping):
                    if i == 0:
                        org_offset = sub_list[1]
                    else:
                        if sub_list[0] != org_offset:
                            last_offset += 1
                        org_offset = sub_list[1]
                    all_offset_mapping += [[last_offset, sub_list[1] - sub_list[0] + last_offset]]
                    last_offset = all_offset_mapping[-1][-1]
                all_doc_tokens += tokenizer.tokenize(this_text_line)

            all_doc_token_boxes = _process_bbox(all_doc_tokens, bbox_lines, all_offset_mapping, 0)

            start_offset = 0
            doc_spans = []
            max_tokens_for_doc = max_seq_len - 2
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
                offset_mapping = [(0, 0)]
                doc_start = doc_span["start"]
                doc_end = doc_span["start"] + doc_span["length"] - 1
                text_offset = all_offset_mapping[doc_start][0]
                text_length = text_offset + all_offset_mapping[doc_end][1]

                for i in range(doc_span["length"]):
                    split_org_index = doc_span["start"] + i
                    tokens.append(all_doc_tokens[split_org_index])
                    token_boxes.append(all_doc_token_boxes[split_org_index])
                    offset_mapping.append(
                        (
                            all_offset_mapping[doc_start + i][0] - text_offset,
                            all_offset_mapping[doc_start + i][1] - text_offset,
                        )
                    )

                tokens.append(tokenizer.sep_token)
                token_boxes.append([0, 0, 0, 0])
                offset_mapping.append((0, 0))

                input_mask = [1] * len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                input_list = {
                    "input_ids": input_ids,
                    "attention_mask": input_mask,
                    "bbox": token_boxes,
                    "image": padded_image,
                    "offset_mapping": offset_mapping,
                    "text": content[text_offset:text_length],
                    "doc_id": doc_id,
                    "doc_offset": text_offset,
                }

                if with_label:
                    ent_labels = []
                    for e in example["entity_list"]:
                        _start, _end = e["start_index"], e["start_index"] + len(e["text"]) - 1
                        start = map_offset(_start, all_offset_mapping)
                        end = map_offset(_end, all_offset_mapping)
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
                        sh = map_offset(_sh, all_offset_mapping)
                        st = map_offset(_st, all_offset_mapping)
                        oh = map_offset(_oh, all_offset_mapping)
                        ot = map_offset(_ot, all_offset_mapping)
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

    @classmethod
    def batch_decode(self, batch_outputs, offset_mappings, texts, doc_offsets, label_maps, with_prob=False):
        if len(batch_outputs) == 1:
            batch_ent_results = []
            for entity_output, offset_mapping, text, doc_offset in zip(
                batch_outputs[0].numpy(),
                offset_mappings,
                texts,
                doc_offsets,
            ):
                entity_output[:, [0, -1]] -= np.inf
                entity_output[:, :, [0, -1]] -= np.inf
                if with_prob:
                    entity_probs = F.softmax(paddle.to_tensor(entity_output, dtype="float64"), axis=1).numpy()

                ent_list = []
                for l, start, end in zip(*np.where(entity_output > 0.0)):
                    start, end = (offset_mapping[start][0], offset_mapping[end][-1])
                    ent = {
                        "text": text[start:end],
                        "type": label_maps["id2entity"][l],
                        "start_index": start + doc_offset,
                    }
                    if with_prob:
                        ent_prob = entity_probs[l, start, end]
                        ent["probability"] = ent_prob
                    ent_list.append(ent)
                batch_ent_results.append(ent_list)
            return [batch_ent_results]
        else:
            batch_ent_results = []
            batch_rel_results = []
            for entity_output, head_output, tail_output, offset_mapping, text, doc_offset in zip(
                batch_outputs[0].numpy(),
                batch_outputs[1].numpy(),
                batch_outputs[2].numpy(),
                offset_mappings,
                texts,
                doc_offsets,
            ):
                entity_output[:, [0, -1]] -= np.inf
                entity_output[:, :, [0, -1]] -= np.inf
                if with_prob:
                    entity_probs = F.softmax(paddle.to_tensor(entity_output, dtype="float64"), axis=1).numpy()
                    head_probs = F.softmax(paddle.to_tensor(head_output, dtype="float64"), axis=1).numpy()
                    tail_probs = F.softmax(paddle.to_tensor(tail_output, dtype="float64"), axis=1).numpy()

                ents = set()
                ent_list = []
                for l, start, end in zip(*np.where(entity_output > 0.0)):
                    ents.add((start, end))
                    start, end = (offset_mapping[start][0], offset_mapping[end][-1])
                    ent = {
                        "text": text[start:end],
                        "type": label_maps["id2entity"][l],
                        "start_index": start + doc_offset,
                    }
                    if with_prob:
                        ent_prob = entity_probs[l, start, end]
                        ent["probability"] = ent_prob
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
                                "subject_start_index": offset_mapping[sh][0] + doc_offset,
                                "object_start_index": offset_mapping[oh][0] + doc_offset,
                            }
                            if with_prob:
                                rel_prob = head_probs[p, sh, oh] * tail_probs[p, st, ot]
                                rel["probability"] = rel_prob
                            rel_list.append(rel)
                batch_rel_results.append(rel_list)
            return [batch_ent_results, batch_rel_results]


def reader(data_path, tokenizer, max_seq_len=512, doc_stride=128, label_maps=None, lang="ch"):
    with open(data_path, "r", encoding="utf-8") as f:
        examples = []
        for line in f:
            example = json.loads(line)
            examples.append(example)

    tokenized_examples = Processor.preprocess_doc(
        examples,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        doc_stride=doc_stride,
        label_maps=label_maps,
        lang=lang,
    )
    for tokenized_example in tokenized_examples:
        yield tokenized_example


def get_eval_golds(data_path):
    golds = {"entity_list": [], "spo_list": []}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            golds["entity_list"].append(example["entity_list"])
            golds["spo_list"].append(example["spo_list"])
    if all([not spo for spo in golds["spo_list"]]):
        golds["spo_list"] = []
    return golds


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
