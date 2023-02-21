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

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import paddle

from paddlenlp.transformers.tokenizer_utils_base import (
    PaddingStrategy,
    PretrainedTokenizerBase,
)

ignore_list = ["offset_mapping", "text", "image", "bbox"]


@dataclass
class DataCollator:
    tokenizer: PretrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    label_maps: Optional[dict] = None

    def __call__(self, features: List[Dict[str, Union[List[int], paddle.Tensor]]]) -> Dict[str, paddle.Tensor]:
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        new_features = [{k: v for k, v in f.items() if k not in ["labels"] + ignore_list} for f in features]

        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
        )

        batch = [paddle.to_tensor(batch[k]) for k in batch.keys()]
        batch.append(paddle.to_tensor([feature["bbox"] for feature in features]))
        batch.append(paddle.to_tensor([feature["image"] for feature in features]))

        if labels is None:  # for test
            batch.append([feature["offset_mapping"] for feature in features])
            batch.append([feature["text"] for feature in features])
            return batch

        bs = batch[0].shape[0]
        # Ensure the dimension is greater or equal to 1
        max_ent_num = max(max([len(lb["ent_labels"]) for lb in labels]), 1)
        num_ents = len(self.label_maps["entity2id"])
        batch_entity_labels = paddle.zeros(shape=[bs, num_ents, max_ent_num, 2], dtype="int64")
        for i, lb in enumerate(labels):
            for eidx, (l, eh, et) in enumerate(lb["ent_labels"]):
                batch_entity_labels[i, l, eidx, :] = paddle.to_tensor([eh, et])

        if not self.label_maps["relation2id"]:
            batch.append([batch_entity_labels])
        else:
            max_spo_num = max(max([len(lb["rel_labels"]) for lb in labels]), 1)
            num_rels = len(self.label_maps["relation2id"])
            batch_head_labels = paddle.zeros(shape=[bs, num_rels, max_spo_num, 2], dtype="int64")
            batch_tail_labels = paddle.zeros(shape=[bs, num_rels, max_spo_num, 2], dtype="int64")

            for i, lb in enumerate(labels):
                for spidx, (sh, st, p, oh, ot) in enumerate(lb["rel_labels"]):
                    batch_head_labels[i, p, spidx, :] = paddle.to_tensor([sh, oh])
                    batch_tail_labels[i, p, spidx, :] = paddle.to_tensor([st, ot])
            batch.append([batch_entity_labels, batch_head_labels, batch_tail_labels])
        return batch
