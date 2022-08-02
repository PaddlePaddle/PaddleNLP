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

from utils import DedupList


def get_eval(all_preds, raw_data, task_type):
    if task_type == "entity_extraction":
        ex, ey, ez = 1e-10, 1e-10, 1e-10
        for ent_preds, data in zip(all_preds, raw_data):
            pred_ent_set = set([tuple(p.values()) for p in ent_preds])
            gold_ent_set = set([tuple(g.values()) for g in data["entity_list"]])
            ex += len(pred_ent_set & gold_ent_set)
            ey += len(pred_ent_set)
            ez += len(gold_ent_set)
        ent_f1 = round(2 * ex / (ey + ez), 5) if ex != 1e-10 else 0.
        ent_precision = round(ex / ey, 5) if ey != 1e-10 else 0.
        ent_recall = round(ex / ez, 5) if ez != 1e-10 else 0.

        return {
            "entity_f1": ent_f1,
            "entity_precision": ent_precision,
            "entity_recall": ent_recall,
        }
    elif task_type in ["relation_extraction", "opinion_extraction"]:
        all_ent_preds, all_rel_preds = all_preds
        ex, ey, ez = 1e-10, 1e-10, 1e-10
        rx, ry, rz = 1e-10, 1e-10, 1e-10

        for ent_preds, rel_preds, raw_data in zip(all_ent_preds, all_rel_preds,
                                                  raw_data):
            pred_ent_set = set([tuple(p.values()) for p in ent_preds])
            gold_ent_set = set(
                [tuple(g.values()) for g in raw_data["entity_list"]])
            pred_rel_set = set([tuple(p.values()) for p in rel_preds])
            if task_type == "opinion_extraction":
                gold_rel_set = set(
                    [tuple(g.values()) for g in raw_data["aso_list"]])
            else:
                gold_rel_set = set(
                    [tuple(g.values()) for g in raw_data["spo_list"]])
            ex += len(pred_ent_set & gold_ent_set)
            ey += len(pred_ent_set)
            ez += len(gold_ent_set)
            rx += len(pred_rel_set & gold_rel_set)
            ry += len(pred_rel_set)
            rz += len(gold_rel_set)
        ent_f1 = round(2 * ex / (ey + ez), 5) if ex != 1e-10 else 0.
        ent_precision = round(ex / ey, 5) if ey != 1e-10 else 0.
        ent_recall = round(ex / ez, 5) if ez != 1e-10 else 0.

        rel_f1 = round(2 * rx / (ry + rz), 5) if rx != 1e-10 else 0.
        rel_precision = round(rx / ry, 5) if ry != 1e-10 else 0.
        rel_recall = round(rx / rz, 5) if rz != 1e-10 else 0.

        return {
            "entity_f1": ent_f1,
            "entity_precision": ent_precision,
            "entity_recall": ent_recall,
            "relation_f1": rel_f1,
            "relation_precision": rel_precision,
            "relation_recall": rel_recall
        }
    elif task_type == "event_extraction":
        ex, ey, ez = 1e-10, 1e-10, 1e-10  # 事件级别
        ax, ay, az = 1e-10, 1e-10, 1e-10  # 论元级别

        for pred_events, data in zip(all_preds, raw_data):
            R, T = DedupList(), DedupList()
            # 事件级别
            for event in pred_events:
                if any([argu[1] == "触发词" for argu in event]):
                    R.append(list(sorted(event)))
            for event in data["event_list"]:
                T.append(list(sorted(event)))
            for event in R:
                if event in T:
                    ex += 1
            ey += len(R)
            ez += len(T)
            # 论元级别
            R, T = DedupList(), DedupList()
            for event in pred_events:
                for argu in event:
                    if argu[1] != "触发词":
                        R.append(argu)
            for event in data["event_list"]:
                for argu in event:
                    if argu[1] != "触发词":
                        T.append(argu)
            for argu in R:
                if argu in T:
                    ax += 1
            ay += len(R)
            az += len(T)

        e_f1, e_pr, e_rc = round(2 * ex / (ey + ez),
                                 5), round(ex / ey, 5), round(ex / ez, 5)
        a_f1, a_pr, a_rc = round(2 * ax / (ay + az),
                                 5), round(ax / ay, 5), round(ax / az, 5)

        return {
            "event_f1": e_f1,
            "event_precision": e_pr,
            "event_recall": e_rc,
            "argument_f1": a_f1,
            "argument_precision": a_pr,
            "argument_recall": a_rc,
        }
