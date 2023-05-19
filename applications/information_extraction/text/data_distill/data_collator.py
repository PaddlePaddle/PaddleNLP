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

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import paddle

from paddlenlp.transformers.tokenizer_utils_base import (
    PaddingStrategy,
    PretrainedTokenizerBase,
)

ignore_list = ["offset_mapping", "text"]


@dataclass
class DataCollator:
    tokenizer: PretrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    label_maps: Optional[dict] = None
    task_type: Optional[str] = None

    def __call__(self, features: List[Dict[str, Union[List[int], paddle.Tensor]]]) -> Dict[str, paddle.Tensor]:
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        new_features = [{k: v for k, v in f.items() if k not in ["labels"] + ignore_list} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
        )

        batch = [paddle.to_tensor(batch[k]) for k in batch.keys()]

        if labels is None:  # for test
            if "offset_mapping" in features[0].keys():
                batch.append([feature["offset_mapping"] for feature in features])
            if "text" in features[0].keys():
                batch.append([feature["text"] for feature in features])
            return batch

        bs = batch[0].shape[0]
        if self.task_type == "entity_extraction":
            # Ensure the dimension is greater or equal to 1
            max_ent_num = max(max([len(lb["ent_labels"]) for lb in labels]), 1)
            num_ents = len(self.label_maps["entity2id"])
            batch_entity_labels = paddle.zeros(shape=[bs, num_ents, max_ent_num, 2], dtype="int64")
            for i, lb in enumerate(labels):
                for eidx, (l, eh, et) in enumerate(lb["ent_labels"]):
                    batch_entity_labels[i, l, eidx, :] = paddle.to_tensor([eh, et])

            batch.append([batch_entity_labels])
        else:
            # Ensure the dimension is greater or equal to 1
            max_ent_num = max(max([len(lb["ent_labels"]) for lb in labels]), 1)
            max_spo_num = max(max([len(lb["rel_labels"]) for lb in labels]), 1)
            num_ents = len(self.label_maps["entity2id"])
            if "relation2id" in self.label_maps.keys():
                num_rels = len(self.label_maps["relation2id"])
            else:
                num_rels = len(self.label_maps["sentiment2id"])
            batch_entity_labels = paddle.zeros(shape=[bs, num_ents, max_ent_num, 2], dtype="int64")
            batch_head_labels = paddle.zeros(shape=[bs, num_rels, max_spo_num, 2], dtype="int64")
            batch_tail_labels = paddle.zeros(shape=[bs, num_rels, max_spo_num, 2], dtype="int64")

            for i, lb in enumerate(labels):
                for eidx, (l, eh, et) in enumerate(lb["ent_labels"]):
                    batch_entity_labels[i, l, eidx, :] = paddle.to_tensor([eh, et])
                for spidx, (sh, st, p, oh, ot) in enumerate(lb["rel_labels"]):
                    batch_head_labels[i, p, spidx, :] = paddle.to_tensor([sh, oh])
                    batch_tail_labels[i, p, spidx, :] = paddle.to_tensor([st, ot])
            batch.append([batch_entity_labels, batch_head_labels, batch_tail_labels])
        return batch
