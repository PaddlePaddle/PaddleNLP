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
from functools import partial

import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer, AutoModel
from paddlenlp.utils.log import logger

from utils import postprocess, create_dataloader, reader
from model import TPLinkerPlus


@paddle.no_grad()
def evaluate(model, data_loader, maps):
    model.eval()
    all_predictions = []
    for batch in tqdm(data_loader, desc="Evaluating: ", leave=False):
        input_ids, attention_masks, offset_mappings, texts = batch
        logits = model(input_ids, attention_masks)
        outputs_gathered = postprocess(logits, offset_mappings, texts,
                                       input_ids.shape[1], maps)
        all_predictions.extend(outputs_gathered)

    X, Y, Z = 1e-10, 1e-10, 1e-10
    for preds, golds in zip(all_predictions, data_loader.dataset.raw_data):
        R = set(preds)
        T = set([tuple(g) for g in golds["spo_list"]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1 = 2 * X / (Y + Z) if X != 1e-10 else 0.
    precision = X / Y if Y != 1e-10 else 0.
    recall = X / Z if Z != 1e-10 else 0.
    model.train()
    return precision, recall, f1


def do_eval():
    rel2id = {}
    id2rel = {}
    with open("data/all_50_schemas", "r", encoding="utf-8") as f:
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

    re_maps = {"rel2id": rel2id, "id2tag": id2tag, "id2rel": id2rel}

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
    encoder = AutoModel.from_pretrained("ernie-3.0-base-zh")
    model = TPLinkerPlus(encoder, rel2id, shaking_type="cln")
    state_dict = paddle.load(
        os.path.join(args.model_path, "model_state.pdparams"))
    model.set_dict(state_dict)

    test_ds = load_dataset(reader, data_path=args.test_path, lazy=False)

    test_dataloader = create_dataloader(test_ds,
                                        tokenizer,
                                        max_seq_len=args.max_seq_len,
                                        batch_size=args.batch_size,
                                        is_train=False,
                                        rel2id=rel2id)

    precision, recall, f1 = evaluate(model, test_dataloader, re_maps)
    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" %
                (precision, recall, f1))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument("--test_path", type=str, default=None, help="The path of test set.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization.")

    args = parser.parse_args()
    # yapf: enable

    do_eval()
