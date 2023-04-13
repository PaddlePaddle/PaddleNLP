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

import argparse

import paddle
from metric import get_eval
from tqdm import tqdm
from utils import (
    ClosedDomainIEProcessor,
    create_dataloader,
    get_eval_golds,
    get_label_maps,
    reader,
)

from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer, ErnieLayoutForClosedDomainIE
from paddlenlp.utils.log import logger


@paddle.no_grad()
def evaluate(model, dataloader, label_maps, golds):
    model.eval()
    all_preds = {
        "entity_preds": [],
        "spo_preds": [],
    }
    cur_doc_id = -1
    for batch in tqdm(dataloader, desc="Evaluating: ", leave=False):
        logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["bbox"],
            batch["image"],
        )
        all_preds, cur_doc_id = ClosedDomainIEProcessor.postprocess(
            logits,
            all_preds,
            batch["doc_id"],
            cur_doc_id,
            batch["offset_mapping"],
            batch["text"],
            batch["text_offset"],
            label_maps,
        )

    eval_results = get_eval(all_preds, golds)
    model.train()
    return eval_results


def do_eval():
    label_maps = get_label_maps(args.label_maps_path)
    golds = get_eval_golds(args.test_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = ErnieLayoutForClosedDomainIE.from_pretrained(args.model_path)

    test_ds = load_dataset(
        reader,
        data_path=args.test_path,
        tokenizer=tokenizer,
        label_maps=label_maps,
        max_seq_len=args.max_seq_len,
        lazy=False,
    )

    test_dataloader = create_dataloader(
        test_ds,
        tokenizer=tokenizer,
        label_maps=label_maps,
        batch_size=args.batch_size,
        mode="test",
    )

    precision, recall, f1 = evaluate(model, test_dataloader, label_maps, golds)
    logger.info("Evaluation Precision： %.5f, Recall: %.5f, F1: %.5f" % (precision, recall, f1))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument("--test_path", type=str, default=None, help="The path of test set.")
    parser.add_argument("--label_maps_path", default="./data/label_maps.json", type=str, help="The file path of the labels dictionary.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--doc_stride", type=int, default=256, help="Window size of sliding window.")

    args = parser.parse_args()
    # yapf: enable

    do_eval()
