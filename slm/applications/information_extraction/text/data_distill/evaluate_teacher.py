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

import argparse

import paddle
from metric import get_eval
from tqdm import tqdm
from utils import create_dataloader, get_label_maps, reader, synthetic2distill

from paddlenlp import Taskflow
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.log import logger


@paddle.no_grad()
def evaluate(uie, dataloader, task_type="relation_extraction"):
    all_preds = ([], []) if task_type in ["opinion_extraction", "relation_extraction", "event_extraction"] else []

    infer_results = []
    all_texts = []
    for batch in tqdm(dataloader, desc="Evaluating: ", leave=False):
        _, _, _, texts = batch
        all_texts.extend(texts)
        infer_results.extend(uie(texts))

    infer_results = synthetic2distill(all_texts, infer_results, task_type)

    for res in infer_results:
        if task_type == "entity_extraction":
            all_preds.append(res["entity_list"])
        else:
            all_preds[0].append(res["entity_list"])
            all_preds[1].append(res["spo_list"])

    eval_results = get_eval(all_preds, dataloader.dataset.raw_data, task_type)
    return eval_results


def do_eval():
    # Load trained UIE model
    uie = Taskflow("information_extraction", schema=args.schema, batch_size=args.batch_size, task_path=args.model_path)

    label_maps = get_label_maps(args.task_type, args.label_maps_path)

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")

    test_ds = load_dataset(reader, data_path=args.test_path, lazy=False)

    test_dataloader = create_dataloader(
        test_ds,
        tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        label_maps=label_maps,
        mode="test",
        task_type=args.task_type,
    )

    eval_result = evaluate(uie, test_dataloader, task_type=args.task_type)
    logger.info("Evaluation precision: " + str(eval_result))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument("--test_path", type=str, default=None, help="The path of test set.")
    parser.add_argument("--label_maps_path", default="./ner_data/label_maps.json", type=str, help="The file path of the labels dictionary.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=256, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--task_type", choices=['relation_extraction', 'event_extraction', 'entity_extraction', 'opinion_extraction'], default="entity_extraction", type=str, help="Select the training task type.")

    args = parser.parse_args()
    # yapf: enable

    schema = {"武器名称": ["产国", "类型", "研发单位"]}

    args.schema = schema

    do_eval()
