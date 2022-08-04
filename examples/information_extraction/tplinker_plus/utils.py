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

import re
import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from itertools import groupby
from tqdm import tqdm

import numpy as np
import paddle
from paddlenlp.transformers.tokenizer_utils_base import PretrainedTokenizerBase, PaddingStrategy

ignore_list = ["offset_mapping", "text"]

criteria_map = {
    "entity_extraction": "entity_f1",
    "opinion_extraction": "relation_f1",  # ASO
    "relation_extraction": "relation_f1",  # SPO
    "event_extraction": "argument_f1"
}


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def reader(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            yield json_line


def process_ee(example):
    events = []
    for e in example["event_list"]:
        events.append(
            [[e["event_type"], "触发词", e["trigger"], e["trigger_start_index"]]])
        for a in e["arguments"]:
            events[-1].append([
                e["event_type"], a["role"], a["argument"],
                a["argument_start_index"]
            ])
    return {"text": example['text'], "event_list": events}


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def get_spots_fr_shaking_tag(shaking_idx2matrix_idx, shaking_outputs):
    """
    shaking_tag -> spots
    shaking_tag: (shaking_seq_len, tag_id)
    spots: [(start_ind, end_ind, tag_id), ]
    """
    spots = []
    pred_shaking_tag = shaking_outputs > 0.
    nonzero_points = paddle.nonzero(pred_shaking_tag, as_tuple=False)
    for point in nonzero_points:
        shaking_idx, tag_idx = point[0].item(), point[1].item()
        pos1, pos2 = shaking_idx2matrix_idx[shaking_idx]
        spot = (pos1, pos2, tag_idx)
        spots.append(spot)
    return spots


def get_label_dict(task_type="relation_extraction", label_dict_path=None):
    if task_type in [
            "relation_extraction", "opinion_extraction", "entity_extraction"
    ]:
        with open(label_dict_path, 'r', encoding='utf-8') as fp:
            label_dict = json.load(fp)

        entity2id = label_dict['entity2id']
        if task_type == "relation_extraction":
            relation2id = label_dict['relation2id']
        elif task_type == "opinion_extraction":
            relation2id = label_dict['sentiment2id']

        tags = []
        if task_type != "entity_extraction":
            id2rel = {idx: t for t, idx in relation2id.items()}
            label_dict['id2rel'] = id2rel
            link_types = [
                "SH2OH",  # subject head to object head
                "OH2SH",  # object head to subject head
                "ST2OT",  # subject tail to object tail
                "OT2ST",  # object tail to subject tail
            ]

            for lk in link_types:
                for rel in relation2id.keys():
                    tags.append("=".join([rel, lk]))

        for ent in entity2id.keys():
            tags.append("=".join([ent, "EH2ET"]))

        tag2id = {t: idx for idx, t in enumerate(tags)}
        id2tag = {idx: t for t, idx in tag2id.items()}

        label_dict['id2tag'] = id2tag
        label_dict['tag2id'] = tag2id
    elif task_type == "event_extraction":
        tag2id = {}
        id2tag = {}
        with open(label_dict_path, "r", encoding="utf-8") as fp:
            label_dict = json.load(fp)

        schemas = label_dict['schema_list']
        for schema in schemas:
            t = schema["event_type"]
            for r in ["触发词"] + [s["role"] for s in schema["role_list"]]:
                id2tag[len(tag2id)] = (t, r)
                tag2id[(t, r)] = len(tag2id)
        id2tag[len(tag2id)] = "SH2OH"
        tag2id["SH2OH"] = len(tag2id)
        id2tag[len(tag2id)] = "ST2OT"
        tag2id["ST2OT"] = len(tag2id)
        label_dict["tag2id"] = tag2id
        label_dict["id2tag"] = id2tag
    return label_dict


@dataclass
class DataCollator:
    tokenizer: PretrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    label_dict: Optional[dict] = None
    task_type: Optional[str] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], paddle.Tensor]]]
    ) -> Dict[str, paddle.Tensor]:
        labels = ([feature["labels"] for feature in features]
                  if "labels" in features[0].keys() else None)
        new_features = [{
            k: v
            for k, v in f.items() if k not in ["labels"] + ignore_list
        } for f in features]
        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
        )

        batch = [paddle.to_tensor(batch[k]) for k in batch.keys()]

        if labels is None:  # for test
            if "offset_mapping" in features[0].keys():
                batch.append(
                    [feature["offset_mapping"] for feature in features])
            if "text" in features[0].keys():
                batch.append([feature["text"] for feature in features])
            return batch

        num_tags = len(self.label_dict['tag2id'])
        bs = batch[0].shape[0]
        seqlen = batch[0].shape[1]
        mask = paddle.triu(paddle.ones(shape=[seqlen, seqlen]), diagonal=0)
        mask = paddle.cast(mask, "bool")

        batch_shaking_tag = paddle.zeros(shape=[bs, seqlen, seqlen, num_tags],
                                         dtype="float64")

        if self.task_type == "event_extraction":
            for i, lb in enumerate(labels):
                # argu_labels
                for argu in lb["argu_labels"]:
                    l = argu[0]
                    a = argu[1:]
                    a = [(a[i * 2], a[i * 2 + 1]) for i in range(len(a) // 2)]
                    for h, t in a:
                        batch_shaking_tag[i, h, t, l] = 1
                # head_labels
                for h1, h2 in lb["head_labels"]:
                    batch_shaking_tag[i, h1, h2, -2] = 1
                # tail_labels
                for t1, t2 in lb["tail_labels"]:
                    batch_shaking_tag[i, t1, t2, -1] = 1
        elif self.task_type in [
                "opinion_extraction", "relation_extraction", "entity_extraction"
        ]:
            if self.task_type == "relation_extraction":
                num_rels = len(self.label_dict["relation2id"])
            elif self.task_type == "opinion_extraction":
                num_rels = len(self.label_dict["sentiment2id"])
            else:
                num_rels = 0

            for i, lb in enumerate(labels):
                for sh, st, p, oh, ot in lb["rel_labels"]:
                    # SH2OH
                    batch_shaking_tag[i, sh, oh, p] = 1
                    # OH2SH
                    batch_shaking_tag[i, oh, sh, p + num_rels] = 1
                    # ST2OT
                    batch_shaking_tag[i, st, ot, p + num_rels * 2] = 1
                    # OT2ST
                    batch_shaking_tag[i, ot, st, p + num_rels * 3] = 1
                for l, eh, et in lb["ent_labels"]:
                    # EH2ET
                    batch_shaking_tag[i, eh, et, l + num_rels * 4] = 1

        mask = mask[None, :, :, None]
        mask = paddle.expand(mask, shape=[bs, seqlen, seqlen, num_tags])
        batch.append(
            batch_shaking_tag.masked_select(mask).reshape([bs, -1, num_tags]))
        return batch


def create_dataloader(dataset,
                      tokenizer,
                      max_seq_len=128,
                      batch_size=1,
                      label_dict=None,
                      mode="train",
                      task_type="relation_extraction"):

    def tokenize_and_align_train_labels(example):
        tokenized_inputs = tokenizer(
            example['text'],
            max_length=max_seq_len,
            padding=False,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_offsets_mapping=True,
        )
        offset_mapping = tokenized_inputs["offset_mapping"]
        if task_type == "event_extraction":
            events = []
            for e in example['event_list']:
                events.append([])
                for t, r, a, i in e:
                    label = label_dict['tag2id'][(t, r)]
                    _start, _end = i, i + len(a) - 1
                    start = map_offset(_start, offset_mapping)
                    end = map_offset(_end, offset_mapping)
                    events[-1].append((label, start, end))

            argu_labels = {}
            head_labels = []
            tail_labels = []
            for e in events:
                for l, h, t in e:
                    if l not in argu_labels:
                        argu_labels[l] = [l]
                    argu_labels[l].extend([h, t])

                for i1, (_, h1, t1) in enumerate(e):
                    for i2, (_, h2, t2) in enumerate(e):
                        if i2 > i1:
                            head_labels.append([min(h1, h2), max(h1, h2)])
                            tail_labels.append([min(t1, t2), max(t1, t2)])
            argu_labels = list(argu_labels.values())

            outputs = {
                "input_ids": tokenized_inputs["input_ids"],
                "attention_mask": tokenized_inputs["attention_mask"],
                "labels": {
                    "argu_labels":
                    argu_labels if len(argu_labels) > 0 else [[0, 0, 0]],
                    "head_labels":
                    head_labels if len(head_labels) > 0 else [[0, 0]],
                    "tail_labels":
                    tail_labels if len(tail_labels) > 0 else [[0, 0]]
                }
            }
        elif task_type in [
                "relation_extraction", "opinion_extraction", "entity_extraction"
        ]:

            ent_labels = []
            for e in example["entity_list"]:
                _start, _end = e['start_index'], e['start_index'] + len(
                    e['text']) - 1
                start = map_offset(_start, offset_mapping)
                end = map_offset(_end, offset_mapping)
                label = label_dict['entity2id'][e['type']]
                ent_labels.append([label, start, end])

            outputs = {
                "input_ids": tokenized_inputs["input_ids"],
                "attention_mask": tokenized_inputs["attention_mask"],
                "labels": {
                    "ent_labels": ent_labels,
                    "rel_labels": []
                }
            }

            if task_type == "relation_extraction":
                rel_labels = []
                for r in example["spo_list"]:
                    _sh, _oh = r["subject_start_index"], r["object_start_index"]
                    _st, _ot = _sh + len(r["subject"]) - 1, _oh + len(
                        r["object"]) - 1
                    sh = map_offset(_sh, offset_mapping)
                    st = map_offset(_st, offset_mapping)
                    oh = map_offset(_oh, offset_mapping)
                    ot = map_offset(_ot, offset_mapping)
                    p = label_dict["relation2id"][r["predicate"]]
                    rel_labels.append([sh, st, p, oh, ot])
                outputs['labels']['rel_labels'] = rel_labels
            elif task_type == "opinion_extraction":
                rel_labels = []
                for r in example["aso_list"]:
                    _ah, _oh = r["aspect_start_index"], r["opinion_start_index"]
                    _at, _ot = _sh + len(r["aspect"]) - 1, _oh + len(
                        r["opinion"]) - 1
                    ah = map_offset(_ah, offset_mapping)
                    at = map_offset(_at, offset_mapping)
                    oh = map_offset(_oh, offset_mapping)
                    ot = map_offset(_ot, offset_mapping)
                    s = label_dict["sentiment2id"][r["sentiment"]]
                    rel_labels.append([ah, at, s, oh, ot])
                outputs['labels']['rel_labels'] = rel_labels
        return outputs

    def tokenize(example):
        tokenized_inputs = tokenizer(
            example['text'],
            max_length=max_seq_len,
            padding=False,
            truncation=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )
        tokenized_inputs['text'] = example['text']
        return tokenized_inputs

    if task_type == "event_extraction" and not mode == "infer":
        dataset.map(process_ee)

    if mode == "train":
        dataset = dataset.map(tokenize_and_align_train_labels)
    else:
        dataset_copy = copy.deepcopy(dataset)
        dataset = dataset.map(tokenize)

    data_collator = DataCollator(tokenizer,
                                 label_dict=label_dict,
                                 task_type=task_type)

    shuffle = True if mode == "train" else False
    batch_sampler = paddle.io.BatchSampler(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle)
    dataloader = paddle.io.DataLoader(dataset=dataset,
                                      batch_sampler=batch_sampler,
                                      collate_fn=data_collator,
                                      num_workers=0,
                                      return_list=True)
    if mode != "train":
        dataloader.dataset.raw_data = dataset_copy
    return dataloader


def isin(event_a, event_b):
    """判断event_a是否event_b的一个子集"""
    if event_a["event_type"] != event_b["event_type"]:
        return False
    for argu in event_a["arguments"]:
        if argu not in event_b["arguments"]:
            return False
    return True


class DedupList(list):
    """定义去重的list"""

    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def neighbors(host, argus, links):
    """构建邻集（host节点与其所有邻居的集合）"""
    results = [host]
    for argu in argus:
        if host[2:] + argu[2:] in links:
            results.append(argu)
    return list(sorted(results))


def clique_search(argus, links):
    """搜索每个节点所属的完全子图作为独立事件
    搜索思路：找出不相邻的节点，然后分别构建它们的邻集，递归处理。
    """
    Argus = DedupList()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links:
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        results = DedupList()
        for A in Argus:
            for a in clique_search(A, links):
                results.append(a)
        return results
    else:
        return [list(sorted(argus))]


def extract_events(pred_events, text):
    event_list = DedupList()
    for event in pred_events:
        final_event = {'event_type': event[0][0], 'arguments': DedupList()}
        for argu in event:
            if argu[1] != u'触发词':
                final_event['arguments'].append({
                    'role': argu[1],
                    'argument': argu[2]
                })
        event_list = [
            event for event in event_list if not isin(event, final_event)
        ]
        if not any([isin(final_event, event) for event in event_list]):
            event_list.append(final_event)
    return event_list


def postprocess(batch_outputs,
                offset_mappings,
                texts,
                seqlen,
                label_dict,
                task_type="relation_extraction"):
    if task_type == "event_extraction":
        batch_results = []
        for shaking_outputs, offset_mapping, text in zip(
                batch_outputs, offset_mappings, texts):
            shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(seqlen)
                                      for end_ind in list(range(seqlen))[ind:]]
            matrix_spots = get_spots_fr_shaking_tag(shaking_idx2matrix_idx,
                                                    shaking_outputs)
            argus = set()
            heads = set()
            tails = set()
            # Token length
            actual_len = len(offset_mapping) - 2
            for sp in matrix_spots:
                tag = label_dict["id2tag"][sp[2]]
                if sp[0] > sp[1] or sp[1] > actual_len:
                    continue
                if tag == "SH2OH":
                    heads.add((sp[0], sp[1]))
                elif tag == "ST2OT":
                    tails.add((sp[0], sp[1]))
                else:
                    argus.add(tag + (sp[0], sp[1]))

            # 构建链接
            links = set()
            for i1, (_, _, h1, t1) in enumerate(argus):
                for i2, (_, _, h2, t2) in enumerate(argus):
                    if i2 > i1:
                        if (min(h1, h2), max(h1, h2)) in heads:
                            if (min(t1, t2), max(t1, t2)) in tails:
                                links.add((h1, t1, h2, t2))
                                links.add((h2, t2, h1, t1))

            # 析出事件
            events = []
            for _, sub_argus in groupby(sorted(argus), key=lambda s: s[0]):
                for event in clique_search(list(sub_argus), links):
                    events.append([])
                    for argu in event:
                        start, end = (
                            offset_mapping[argu[2]][0],
                            offset_mapping[argu[3]][1],
                        )
                        events[-1].append(
                            [argu[0], argu[1], text[start:end], start])
                    if all([argu[1] != "触发词" for argu in event]):
                        events.pop()

            batch_results.append(events)
        return batch_results
    elif task_type in [
            "opinion_extraction", "relation_extraction", "entity_extraction"
    ]:
        batch_ent_results = []
        batch_rel_results = []
        for shaking_outputs, offset_mapping, text in zip(
                batch_outputs, offset_mappings, texts):
            shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(seqlen)
                                      for end_ind in list(range(seqlen))[ind:]]
            matrix_spots = get_spots_fr_shaking_tag(shaking_idx2matrix_idx,
                                                    shaking_outputs)
            head_ind2entities = {}
            ent_list = []
            # Token length
            actual_len = len(offset_mapping) - 2
            for sp in matrix_spots:
                tag = label_dict["id2tag"][sp[2]]
                ent_type, link_type = tag.split("=")
                # For an entity, the start position can not be larger than the end pos.
                if link_type != "EH2ET" or sp[0] > sp[1] or sp[1] > actual_len:
                    continue

                start, end = (offset_mapping[sp[0]][0],
                              offset_mapping[sp[1]][1])

                ent = {
                    "text": text[start:end],
                    "type": ent_type,
                    "start_index": start
                }
                ent_list.append(ent)
                # Take ent_head_pos as the key to entity list
                head_ind2entities.setdefault(start, []).append(ent)

            batch_ent_results.append(ent_list)

            if task_type != "entity_extraction":
                rel_list = []
                tail_link_memory_set = set()
                for sp in matrix_spots:
                    tag = label_dict["id2tag"][sp[2]]
                    _, link_type = tag.split("=")

                    rel_id = 0
                    if link_type == "ST2OT" and sp[0] <= actual_len and sp[
                            1] <= actual_len:
                        subj_tail = offset_mapping[sp[0]][1] - 1
                        obj_tail = offset_mapping[sp[1]][1] - 1
                        tail_link_memory = (rel_id, subj_tail, obj_tail)
                        tail_link_memory_set.add(tail_link_memory)
                    elif link_type == "OT2ST" and sp[0] <= actual_len and sp[
                            1] <= actual_len:
                        subj_tail = offset_mapping[sp[1]][1] - 1
                        obj_tail = offset_mapping[sp[0]][1] - 1
                        tail_link_memory = (rel_id, subj_tail, obj_tail)
                        tail_link_memory_set.add(tail_link_memory)

                # Head link
                for sp in matrix_spots:
                    tag = label_dict["id2tag"][sp[2]]
                    rel_label, link_type = tag.split("=")

                    if link_type == "SH2OH" and sp[0] <= actual_len and sp[
                            1] <= actual_len:
                        _subj_head, _obj_head = sp[0], sp[1]
                    elif link_type == "OH2SH" and sp[0] <= actual_len and sp[
                            1] <= actual_len:
                        _subj_head, _obj_head = sp[1], sp[0]
                    else:
                        continue

                    subj_head_key = offset_mapping[_subj_head][0]
                    obj_head_key = offset_mapping[_obj_head][0]

                    if (subj_head_key not in head_ind2entities
                            or obj_head_key not in head_ind2entities):
                        # No entity start with subj_head_key and obj_head_key
                        continue

                    # All entities start with this subject head
                    subj_list = head_ind2entities[subj_head_key]
                    # All entities start with this object head
                    obj_list = head_ind2entities[obj_head_key]

                    # Go over all subj-obj pair to check whether the tail link exists
                    for subj in subj_list:
                        for obj in obj_list:
                            subj_tail = subj["start_index"] + len(
                                subj["text"]) - 1
                            obj_tail = obj["start_index"] + len(obj["text"]) - 1
                            tail_link_memory = (rel_id, subj_tail, obj_tail)

                            if tail_link_memory not in tail_link_memory_set:
                                continue

                            if task_type == "relation_extraction":
                                rel = {
                                    "subject": subj["text"],
                                    "predicate": rel_label,
                                    "object": obj["text"],
                                    "subject_start_index": subj["start_index"],
                                    "object_start_index": obj["start_index"]
                                }
                            else:
                                rel = {
                                    "aspect": subj["text"],
                                    "sentiment": rel_label,
                                    "opinion": obj["text"],
                                    "aspect_start_index": subj["start_index"],
                                    "opinion_start_index": obj["start_index"]
                                }
                            rel_list.append(rel)
                batch_rel_results.append(rel_list)
        if task_type == "entity_extraction":
            return batch_ent_results
        else:
            return (batch_ent_results, batch_rel_results)
