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


def get_eval(all_preds, golds):
    all_ent_preds, all_rel_preds = all_preds["entity_preds"], all_preds["spo_preds"]
    ent_golds, rel_golds = golds["entity_list"], golds["spo_list"]

    ex, ey, ez = 0, 0, 0
    for ent_preds, ent_gold in zip(all_ent_preds, ent_golds):
        pred_ent_set = set([tuple(p.values()) for p in ent_preds])
        gold_ent_set = set([tuple(g.values()) for g in ent_gold])
        ex += len(pred_ent_set & gold_ent_set)
        ey += len(pred_ent_set)
        ez += len(gold_ent_set)

    rx, ry, rz = 0, 0, 0
    for rel_preds, rel_gold in zip(all_rel_preds, rel_golds):
        pred_rel_set = set([tuple(p.values()) for p in rel_preds])
        gold_rel_set = set([tuple(g.values()) for g in rel_gold])
        rx += len(pred_rel_set & gold_rel_set)
        ry += len(pred_rel_set)
        rz += len(gold_rel_set)

    precision = round((ex + rx) / (ey + ry), 5) if (ey + ry) > 0 else 0.0
    recall = round((ex + rx) / (ez + rz), 5) if (ez + rz) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 5) if (precision + recall) > 0 else 0.0

    return precision, recall, f1
