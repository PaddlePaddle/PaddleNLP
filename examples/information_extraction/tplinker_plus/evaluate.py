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

import argparse
import json
import os
from tqdm import tqdm

import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer, AutoModel
from paddlenlp.utils.log import logger

from utils import postprocess, create_dataloader, reader, get_label_dict, DedupList
from model import TPLinkerPlus


@paddle.no_grad()
def evaluate(model, dataloader, label_dicts, task_type="relation_extraction"):
    model.eval()
    if task_type == "event_extraction":
        all_predictions = []
        for batch in tqdm(dataloader, desc="Evaluating: ", leave=False):
            input_ids, attention_masks, offset_mappings, texts = batch
            logits = model(input_ids, attention_masks)
            outputs_gathered = postprocess(logits, offset_mappings, texts,
                                           input_ids.shape[1], label_dicts,
                                           task_type)
            all_predictions.extend(outputs_gathered)

        ex, ey, ez = 1e-10, 1e-10, 1e-10  # 事件级别
        ax, ay, az = 1e-10, 1e-10, 1e-10  # 论元级别

        for pred_events, raw_data in zip(all_predictions,
                                         dataloader.dataset.raw_data):
            R, T = DedupList(), DedupList()
            # 事件级别
            for event in pred_events:
                if any([argu[1] == "触发词" for argu in event]):
                    R.append(list(sorted(event)))
            for event in raw_data["event_list"]:
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
            for event in raw_data["event_list"]:
                for argu in event:
                    if argu[1] != "触发词":
                        T.append(argu)
            for argu in R:
                if argu in T:
                    ax += 1
            ay += len(R)
            az += len(T)

        e_f1, e_pr, e_rc = 2 * ex / (ey + ez), ex / ey, ex / ez
        a_f1, a_pr, a_rc = 2 * ax / (ay + az), ax / ay, ax / az

        model.train()

        return {
            "event_f1": e_f1,
            "event_precision": e_pr,
            "event_recall": e_rc,
            "argument_f1": a_f1,
            "argument_precision": a_pr,
            "argument_recall": a_rc,
        }
    elif task_type in ["opinion_extraction", "relation_extraction"]:
        all_ent_predictions = []
        all_rel_predictions = []
        for batch in tqdm(dataloader, desc="Evaluating: ", leave=False):
            input_ids, attention_masks, offset_mappings, texts = batch
            logits = model(input_ids, attention_masks)
            ent_outputs, rel_outputs = postprocess(logits, offset_mappings,
                                                   texts, input_ids.shape[1],
                                                   label_dicts, task_type)
            all_ent_predictions.extend(ent_outputs)
            all_rel_predictions.extend(rel_outputs)

        ex, ey, ez = 1e-10, 1e-10, 1e-10
        rx, ry, rz = 1e-10, 1e-10, 1e-10

        for ent_preds, rel_preds, raw_data in zip(all_ent_predictions,
                                                  all_rel_predictions,
                                                  dataloader.dataset.raw_data):
            pred_ent_set = set([tuple(p.values()) for p in ent_preds])
            gold_ent_set = set(
                [tuple(g.values()) for g in raw_data["entity_list"]])
            pred_rel_set = set([tuple(p.values()) for p in rel_preds])
            gold_rel_set = set(
                [tuple(g.values()) for g in raw_data["relation_list"]])
            ex += len(pred_ent_set & gold_ent_set)
            ey += len(pred_ent_set)
            ez += len(gold_ent_set)
            rx += len(pred_rel_set & gold_rel_set)
            ry += len(pred_rel_set)
            rz += len(gold_rel_set)
        ent_f1 = 2 * ex / (ey + ez) if ex != 1e-10 else 0.
        ent_precision = ex / ey if ey != 1e-10 else 0.
        ent_recall = ex / ez if ez != 1e-10 else 0.

        rel_f1 = 2 * rx / (ry + rz) if rx != 1e-10 else 0.
        rel_precision = rx / ry if ry != 1e-10 else 0.
        rel_recall = rx / rz if rz != 1e-10 else 0.

        model.train()
        return {
            "entity_f1": ent_f1,
            "entity_precision": ent_precision,
            "entity_recall": ent_recall,
            "relation_f1": rel_f1,
            "relation_precision": rel_precision,
            "relation_recall": rel_recall
        }


def do_eval():
    label_dicts = get_label_dict(args.task_type, args.label_dicts_path)
    num_tags = len(label_dicts["id2tag"])

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
    encoder = AutoModel.from_pretrained("ernie-3.0-base-zh")
    model = TPLinkerPlus(encoder, num_tags, shaking_type="cln")
    state_dict = paddle.load(
        os.path.join(args.model_path, "model_state.pdparams"))
    model.set_dict(state_dict)

    test_ds = load_dataset(reader, data_path=args.test_path, lazy=False)

    test_dataloader = create_dataloader(test_ds,
                                        tokenizer,
                                        max_seq_len=args.max_seq_len,
                                        batch_size=args.batch_size,
                                        is_train=False,
                                        label_dicts=label_dicts,
                                        task_type=args.task_type)

    eval_result = evaluate(model,
                           test_dataloader,
                           label_dicts,
                           task_type=args.task_type)
    logger.info("Evaluation precision: " + str(eval_result))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument("--test_path", type=str, default=None, help="The path of test set.")
    parser.add_argument("--label_dicts_path", default="./duie/label_dicts.json", type=str, help="The file path of the schema for extraction.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--task_type", choices=['relation_extraction', 'event_extraction', 'entity_extraction', 'opinion_extraction'], default="relation_extraction", type=str, help="Select the training task type.")

    args = parser.parse_args()
    # yapf: enable

    do_eval()
