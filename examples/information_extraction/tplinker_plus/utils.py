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
from tqdm import tqdm

import numpy as np
import paddle
from paddlenlp.transformers.tokenizer_utils_base import PretrainedTokenizerBase, PaddingStrategy

ignore_list = ["offset_mapping", "text"]


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def reader(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            yield json_line


def process_train(ds, rel2id):

    def search(pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    def judge(example):
        spo_list = []
        for spo in example["spo_list"]:
            sub = search(spo["subject"], example["text"])
            obj = search(spo["object"], example["text"])
            if sub == -1 or obj == -1:
                continue
            else:
                spo_list.append([1])
        return len(spo_list) > 0

    def convert(example):
        spo_list = []
        for spo in example["spo_list"]:
            sub = search(spo["subject"], example["text"])
            pre = rel2id[spo["predicate"]]
            obj = search(spo["object"], example["text"])
            if sub == -1 or obj == -1:
                continue
            else:
                spo_list.append([
                    sub,
                    sub + len(spo["subject"]) - 1,
                    pre,
                    obj,
                    obj + len(spo["object"]) - 1,
                ])

        assert len(spo_list) > 0
        return {"text": example["text"], "spo_list": spo_list}

    return ds.filter(judge).map(convert)


def process_dev(example):
    triplet = []
    for spo in example["spo_list"]:
        triplet.append([
            spo["subject"],
            spo["predicate"],
            spo["object"],
        ])
    return {"text": example['text'], "spo_list": triplet}


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def create_dataloader(dataset,
                      tokenizer,
                      max_seq_len=128,
                      batch_size=1,
                      is_train=True,
                      rel2id=None):

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
        labels = []
        for spo_list in example["spo_list"]:
            _sh, _st, p, _oh, _ot = spo_list
            sh = map_offset(_sh, offset_mapping)
            st = map_offset(_st, offset_mapping)
            oh = map_offset(_oh, offset_mapping)
            ot = map_offset(_ot, offset_mapping)
            labels.append([sh, st, p, oh, ot])
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": labels
        }

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

    if is_train:
        dataset = process_train(dataset, rel2id=rel2id)
        dataset = dataset.map(tokenize_and_align_train_labels)
    else:
        dataset = dataset.map(process_dev)
        dataset_copy = copy.deepcopy(dataset)
        dataset = dataset.map(tokenize)

    data_collator = DataCollator(tokenizer, num_labels=len(rel2id))

    shuffle = True if is_train else False
    batch_sampler = paddle.io.BatchSampler(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle)
    dataloader = paddle.io.DataLoader(dataset=dataset,
                                      batch_sampler=batch_sampler,
                                      collate_fn=data_collator,
                                      num_workers=0,
                                      return_list=True)
    if not is_train:
        dataloader.dataset.raw_data = dataset_copy
    return dataloader


@dataclass
class DataCollator:
    tokenizer: PretrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    num_labels: Optional[int] = None

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

        if labels is None:  # for test
            batch = [paddle.to_tensor(batch[k]) for k in batch.keys()]
            if "offset_mapping" in features[0].keys():
                batch.append(
                    [feature["offset_mapping"] for feature in features])
            if "text" in features[0].keys():
                batch.append([feature["text"] for feature in features])
            return batch

        for k in batch.keys():
            batch[k] = paddle.to_tensor(batch[k])

        bs = batch["input_ids"].shape[0]
        seqlen = batch["input_ids"].shape[1]
        mask = paddle.triu(paddle.ones(shape=[seqlen, seqlen]), diagonal=0)
        mask = paddle.cast(mask, "bool")

        num_tag = self.num_labels * 4 + 1
        batch_shaking_tag = paddle.zeros(shape=[bs, seqlen, seqlen, num_tag],
                                         dtype="float64")

        for i, lb in enumerate(labels):
            for sh, st, p, oh, ot in lb:
                # SH2OH
                batch_shaking_tag[i, sh, oh, p] = 1
                # OH2SH
                batch_shaking_tag[i, oh, sh, p + self.num_labels] = 1
                # ST2OT
                batch_shaking_tag[i, st, ot, p + self.num_labels * 2] = 1
                # OT2ST
                batch_shaking_tag[i, ot, st, p + self.num_labels * 3] = 1
                # EH2ET
                batch_shaking_tag[i, sh, st, -1] = 1
                batch_shaking_tag[i, oh, ot, -1] = 1
        mask = mask[None, :, :, None]
        mask = paddle.expand(mask, shape=[bs, seqlen, seqlen, num_tag])
        batch["labels"] = batch_shaking_tag.masked_select(mask).reshape(
            [bs, -1, num_tag])
        # batch = [np.array(batch[k], dtype="int64") for k in batch.keys()]
        batch = [paddle.to_tensor(batch[k]) for k in batch.keys()]
        return batch


def postprocess(batch_outputs, offset_mappings, texts, seqlen, maps):
    batch_results = []
    for shaking_outputs, offset_mapping, text in zip(batch_outputs,
                                                     offset_mappings, texts):
        shaking_idx2matrix_idx = [(ind, end_ind) for ind in range(seqlen)
                                  for end_ind in list(range(seqlen))[ind:]]
        head_ind2entities = {}
        rel_list = []

        matrix_spots = get_spots_fr_shaking_tag(shaking_idx2matrix_idx,
                                                shaking_outputs)
        # Token length
        actual_len = len(offset_mapping) - 2
        for sp in matrix_spots:
            tag = maps["id2tag"][sp[2]]
            ent_type, link_type = tag.split("=")
            # For an entity, the start position can not be larger than the end pos.
            if link_type != "EH2ET" or sp[0] > sp[1] or sp[1] > actual_len:
                continue

            entity = {
                "type": ent_type,
                "tok_span": [sp[0], sp[1]],
            }
            # Take ent_head_pos as the key to entity list
            head_key = sp[0]
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append(entity)

        # Tail link
        tail_link_memory_set = set()
        for sp in matrix_spots:
            tag = maps["id2tag"][sp[2]]
            rel, link_type = tag.split("=")

            if link_type == "ST2OT":
                rel = maps["rel2id"][rel]
                tail_link_memory = (rel, sp[0], sp[1])
                tail_link_memory_set.add(tail_link_memory)
            elif link_type == "OT2ST":
                rel = maps["rel2id"][rel]
                tail_link_memory = (rel, sp[1], sp[0])
                tail_link_memory_set.add(tail_link_memory)

        # Head link
        for sp in matrix_spots:
            tag = maps["id2tag"][sp[2]]
            rel, link_type = tag.split("=")

            if link_type == "SH2OH":
                rel = maps["rel2id"][rel]
                subj_head_key, obj_head_key = sp[0], sp[1]
            elif link_type == "OH2SH":
                rel = maps["rel2id"][rel]
                subj_head_key, obj_head_key = sp[1], sp[0]
            else:
                continue

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
                    tail_link_memory = (rel, subj["tok_span"][1],
                                        obj["tok_span"][1])

                    if tail_link_memory not in tail_link_memory_set:
                        continue
                    rel_list.append((
                        text[offset_mapping[subj["tok_span"][0]][0]:
                             offset_mapping[subj["tok_span"][1]][1]],
                        maps["id2rel"][rel],
                        text[offset_mapping[obj["tok_span"][0]][0]:
                             offset_mapping[obj["tok_span"][1]][1]],
                    ))

        batch_results.append(rel_list)
    return batch_results


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


def get_re_label_dict(schema_file_path):
    """
    'rel2id', 'id2tag', 'id2rel'
    """
    rel2id = {}
    id2rel = {}
    with open(schema_file_path, "r", encoding="utf-8") as f:
        for l in f:
            l = json.loads(l)
            if l["predicate"] not in rel2id:
                id2rel[len(rel2id)] = l["predicate"]
                rel2id[l["predicate"]] = len(rel2id)
    link_types = [
        "SH2OH",  # subject head to object head
        "OH2SH",  # object head to subject head
        "ST2OT",  # subject tail to object tail
        "OT2ST",  # object tail to subject tail
    ]
    tags = []
    for lk in link_types:
        for rel in rel2id.keys():
            tags.append("=".join([rel, lk]))
    tags.append("DEFAULT=EH2ET")
    tag2id = {t: idx for idx, t in enumerate(tags)}
    id2tag = {idx: t for t, idx in tag2id.items()}

    label_dicts = {"rel2id": rel2id, "id2tag": id2tag, "id2rel": id2rel}
    return label_dicts