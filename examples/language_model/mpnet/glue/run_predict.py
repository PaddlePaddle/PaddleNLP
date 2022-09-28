# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from functools import partial
import os
import paddle
from paddle.io import DataLoader
import pandas as pd
from tqdm import tqdm
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Tuple, Pad
from paddlenlp.transformers import MPNetForSequenceClassification, MPNetTokenizer
from run_glue import convert_example

task2filename = {
    "cola": "CoLA.tsv",
    "sst-2": "SST-2.tsv",
    "mrpc": "MRPC.tsv",
    "sts-b": "STS-B.tsv",
    "qqp": "QQP.tsv",
    "mnli": ["MNLI-m.tsv", "MNLI-mm.tsv"],
    "rte": "RTE.tsv",
    "qnli": "QNLI.tsv",
    "wnli": "WNLI.tsv"
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        choices=[
            "cola", "sst-2", "mrpc", "sts-b", "qqp", "mnli", "rte", "qnli",
            "wnli"
        ],
        default="cola",
        required=True,
        help="task_name.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    args = parser.parse_args()
    args.task_name = args.task_name.lower()
    return args


def predict(data_loader, model, id2label=None):
    outputs = []
    progress_bar = tqdm(
        range(len(data_loader)),
        desc="Predition Iteration",
    )
    with paddle.no_grad():
        for batch in data_loader:
            input_ids, segment_ids = batch
            logits = model(input_ids)
            if id2label is not None:
                pred = paddle.argmax(logits, axis=-1).cpu().tolist()
                outputs.extend(list(map(lambda x: id2label[x], pred)))
            else:
                pred = logits.squeeze(-1).cpu().tolist()
                outputs.extend(pred)
            progress_bar.update(1)
    return outputs


def writetsv(outputs, file):
    d = {"index": list(range(len(outputs))), "prediction": outputs}
    pd.DataFrame(d).to_csv(file, sep="\t", index=False)
    print(f"Save to {file}.")


def predict2file(args):
    if args.task_name == "mnli":
        test_ds_matched, test_ds_mismatched = load_dataset(
            "glue", "mnli", splits=["test_matched", "test_mismatched"])
        id2label = dict(
            zip(range(len(test_ds_matched.label_list)),
                test_ds_matched.label_list))
    else:
        test_ds = load_dataset("glue", args.task_name, splits="test")
        if test_ds.label_list is not None:
            id2label = dict(
                zip(range(len(test_ds.label_list)), test_ds.label_list))
        else:
            id2label = None

    model = MPNetForSequenceClassification.from_pretrained(args.ckpt_path)
    model.eval()
    tokenizer = MPNetTokenizer.from_pretrained(args.ckpt_path)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    ): fn(samples)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=None,
        max_seq_length=args.max_seq_length,
        is_test=True,
    )

    if args.task_name == "mnli":
        test_ds_matched = test_ds_matched.map(trans_func, lazy=True)
        test_ds_mismatched = test_ds_mismatched.map(trans_func, lazy=True)
        test_batch_sampler_matched = paddle.io.BatchSampler(
            test_ds_matched, batch_size=args.batch_size, shuffle=False)
        test_data_loader_matched = DataLoader(
            dataset=test_ds_matched,
            batch_sampler=test_batch_sampler_matched,
            collate_fn=batchify_fn,
            num_workers=2,
            return_list=True,
        )
        test_batch_sampler_mismatched = paddle.io.BatchSampler(
            test_ds_mismatched, batch_size=args.batch_size, shuffle=False)
        test_data_loader_mismatched = DataLoader(
            dataset=test_ds_mismatched,
            batch_sampler=test_batch_sampler_mismatched,
            collate_fn=batchify_fn,
            num_workers=2,
            return_list=True,
        )
        file_m = os.path.join("template", task2filename[args.task_name][0])
        file_mm = os.path.join("template", task2filename[args.task_name][1])
        matched_outputs = predict(test_data_loader_matched, model, id2label)
        mismatched_outputs = predict(test_data_loader_mismatched, model,
                                     id2label)
        writetsv(matched_outputs, file_m)
        writetsv(mismatched_outputs, file_mm)
    else:
        test_ds = test_ds.map(trans_func, lazy=True)
        test_batch_sampler = paddle.io.BatchSampler(test_ds,
                                                    batch_size=args.batch_size,
                                                    shuffle=False)
        test_data_loader = DataLoader(
            dataset=test_ds,
            batch_sampler=test_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=2,
            return_list=True,
        )
        predict_outputs = predict(test_data_loader, model, id2label)

        file = os.path.join("template", task2filename[args.task_name])
        writetsv(predict_outputs, file)


if __name__ == "__main__":
    args = get_args()
    os.makedirs("template", exist_ok=True)
    predict2file(args)
