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
from itertools import groupby
from tqdm import tqdm

import numpy as np
import paddle

from data_collator import DataCollator

criteria_map = {
    "entity_extraction": "entity_f1",
    "opinion_extraction": "relation_f1",  # (Aspect, Sentiment, Opinion)
    "relation_extraction": "relation_f1",  # (Subject, Predicate, Object)
    "event_extraction": "relation_f1"  # (Trigger, Role, Argument)
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


def get_label_maps(task_type="relation_extraction", label_maps_path=None):
    with open(label_maps_path, 'r', encoding='utf-8') as fp:
        label_maps = json.load(fp)
    if task_type == "entity_extraction":
        entity2id = label_maps['entity2id']
        id2entity = {idx: t for t, idx in entity2id.items()}
        label_maps['id2entity'] = id2entity
    else:
        entity2id = label_maps['entity2id']
        relation2id = label_maps['relation2id'] if task_type in [
            "relation_extraction", "event_extraction"
        ] else label_maps['sentiment2id']
        id2entity = {idx: t for t, idx in entity2id.items()}
        id2relation = {idx: t for t, idx in relation2id.items()}
        label_maps['id2entity'] = id2entity
        label_maps['id2relation'] = id2relation
    return label_maps


def create_dataloader(dataset,
                      tokenizer,
                      max_seq_len=128,
                      batch_size=1,
                      label_maps=None,
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

        ent_labels = []
        for e in example["entity_list"]:
            _start, _end = e['start_index'], e['start_index'] + len(
                e['text']) - 1
            start = map_offset(_start, offset_mapping)
            end = map_offset(_end, offset_mapping)
            if start == -1 or end == -1:
                continue
            label = label_maps['entity2id'][e['type']]
            ent_labels.append([label, start, end])

        outputs = {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": {
                "ent_labels": ent_labels,
                "rel_labels": []
            }
        }

        if task_type in ["relation_extraction", "event_extraction"]:
            rel_labels = []
            for r in example["spo_list"]:
                _sh, _oh = r["subject_start_index"], r["object_start_index"]
                _st, _ot = _sh + len(r["subject"]) - 1, _oh + len(
                    r["object"]) - 1
                sh = map_offset(_sh, offset_mapping)
                st = map_offset(_st, offset_mapping)
                oh = map_offset(_oh, offset_mapping)
                ot = map_offset(_ot, offset_mapping)
                if sh == -1 or st == -1 or oh == -1 or ot == -1:
                    continue
                p = label_maps["relation2id"][r["predicate"]]
                rel_labels.append([sh, st, p, oh, ot])
            outputs['labels']['rel_labels'] = rel_labels
        elif task_type == "opinion_extraction":
            rel_labels = []
            for r in example["aso_list"]:
                _ah, _oh = r["aspect_start_index"], r["opinion_start_index"]
                _at, _ot = _ah + len(r["aspect"]) - 1, _oh + len(
                    r["opinion"]) - 1
                ah = map_offset(_ah, offset_mapping)
                at = map_offset(_at, offset_mapping)
                oh = map_offset(_oh, offset_mapping)
                ot = map_offset(_ot, offset_mapping)
                if ah == -1 or at == -1 or oh == -1 or ot == -1:
                    continue

                s = label_maps["sentiment2id"][r["sentiment"]]
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

    if mode == "train":
        dataset = dataset.map(tokenize_and_align_train_labels)
    else:
        dataset_copy = copy.deepcopy(dataset)
        dataset = dataset.map(tokenize)

    data_collator = DataCollator(tokenizer,
                                 label_maps=label_maps,
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


def postprocess(batch_outputs,
                offset_mappings,
                texts,
                label_maps,
                task_type="relation_extraction"):
    if task_type == "entity_extraction":
        batch_ent_results = []
        for entity_output, offset_mapping, text in zip(batch_outputs[0].numpy(),
                                                       offset_mappings, texts):
            entity_output[:, [0, -1]] -= np.inf
            entity_output[:, :, [0, -1]] -= np.inf
            ent_list = []
            for l, start, end in zip(*np.where(entity_output > 0.)):
                start, end = (offset_mapping[start][0], offset_mapping[end][-1])
                ent = {
                    "text": text[start:end],
                    "type": label_maps['id2entity'][l],
                    "start_index": start
                }
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
            for l, start, end in zip(*np.where(entity_output > 0.)):
                ents.add((start, end))
                start, end = (offset_mapping[start][0], offset_mapping[end][-1])
                ent = {
                    "text": text[start:end],
                    "type": label_maps['id2entity'][l],
                    "start_index": start
                }
                ent_list.append(ent)
            batch_ent_results.append(ent_list)

            rel_list = []
            for sh, st in ents:
                for oh, ot in ents:
                    p1s = np.where(head_output[:, sh, oh] > 0.)[0]
                    p2s = np.where(tail_output[:, st, ot] > 0.)[0]
                    ps = set(p1s) & set(p2s)
                    for p in ps:
                        if task_type in [
                                "relation_extraction", "event_extraction"
                        ]:
                            rel = {
                                "subject":
                                text[offset_mapping[sh][0]:offset_mapping[st]
                                     [1]],
                                "predicate":
                                label_maps['id2relation'][p],
                                "object":
                                text[offset_mapping[oh][0]:offset_mapping[ot]
                                     [1]],
                                "subject_start_index":
                                offset_mapping[sh][0],
                                "object_start_index":
                                offset_mapping[oh][0]
                            }
                        else:
                            rel = {
                                "aspect":
                                text[offset_mapping[sh][0]:offset_mapping[st]
                                     [1]],
                                "sentiment":
                                label_maps['id2relation'][p],
                                "opinion":
                                text[offset_mapping[oh][0]:offset_mapping[ot]
                                     [1]],
                                "aspect_start_index":
                                offset_mapping[sh][0],
                                "opinion_start_index":
                                offset_mapping[oh][0]
                            }
                        rel_list.append(rel)
            batch_rel_results.append(rel_list)
        return (batch_ent_results, batch_rel_results)
