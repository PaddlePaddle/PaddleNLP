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


def get_eval(all_preds, raw_data, task_type):
    if task_type == "entity_extraction":
        ex, ey, ez = 1e-10, 1e-10, 1e-10
        for ent_preds, data in zip(all_preds, raw_data):
            pred_ent_set = set([tuple(p.values()) for p in ent_preds])
            gold_ent_set = set([tuple(g.values()) for g in data["entity_list"]])
            ex += len(pred_ent_set & gold_ent_set)
            ey += len(pred_ent_set)
            ez += len(gold_ent_set)
        ent_f1 = round(2 * ex / (ey + ez), 5) if ex != 1e-10 else 0.0
        ent_precision = round(ex / ey, 5) if ey != 1e-10 else 0.0
        ent_recall = round(ex / ez, 5) if ez != 1e-10 else 0.0

        return {
            "entity_f1": ent_f1,
            "entity_precision": ent_precision,
            "entity_recall": ent_recall,
        }
    else:
        all_ent_preds, all_rel_preds = all_preds

        ex, ey, ez = 1e-10, 1e-10, 1e-10
        for ent_preds, data in zip(all_ent_preds, raw_data):
            pred_ent_set = set([tuple(p.values()) for p in ent_preds])
            gold_ent_set = set([tuple(g.values()) for g in data["entity_list"]])
            ex += len(pred_ent_set & gold_ent_set)
            ey += len(pred_ent_set)
            ez += len(gold_ent_set)
        ent_f1 = round(2 * ex / (ey + ez), 5) if ex != 1e-10 else 0.0
        ent_precision = round(ex / ey, 5) if ey != 1e-10 else 0.0
        ent_recall = round(ex / ez, 5) if ez != 1e-10 else 0.0

        rx, ry, rz = 1e-10, 1e-10, 1e-10

        for rel_preds, raw_data in zip(all_rel_preds, raw_data):
            pred_rel_set = set([tuple(p.values()) for p in rel_preds])
            if task_type == "opinion_extraction":
                gold_rel_set = set([tuple(g.values()) for g in raw_data["aso_list"]])
            else:
                gold_rel_set = set([tuple(g.values()) for g in raw_data["spo_list"]])
            rx += len(pred_rel_set & gold_rel_set)
            ry += len(pred_rel_set)
            rz += len(gold_rel_set)

        rel_f1 = round(2 * rx / (ry + rz), 5) if rx != 1e-10 else 0.0
        rel_precision = round(rx / ry, 5) if ry != 1e-10 else 0.0
        rel_recall = round(rx / rz, 5) if rz != 1e-10 else 0.0

        return {
            "entity_f1": ent_f1,
            "entity_precision": ent_precision,
            "entity_recall": ent_recall,
            "relation_f1": rel_f1,
            "relation_precision": rel_precision,
            "relation_recall": rel_recall,
        }
