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
from functools import partial
import numpy as np

import paddle
from paddle.io import BatchSampler, DataLoader
from paddlenlp.transformers import XLMForSequenceClassification, XLMTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddle.metric import Accuracy

all_languages = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr",
    "ur", "vi", "zh"
]


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to pre-trained model.")
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU/XPU for training.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    args = parser.parse_args()
    return args


@paddle.no_grad()
def evaluate(model, metric, data_loader, language, tokenizer):
    metric.reset()
    for batch in data_loader:
        input_ids, attention_mask, labels = batch
        # add lang_ids
        lang_ids = paddle.ones_like(input_ids) * tokenizer.lang2id[language]
        logits = model(input_ids, langs=lang_ids, attention_mask=attention_mask)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    print("[%s] acc: %s " % (language.upper(), res))
    return res


def convert_example(example, tokenizer, max_seq_length=256, language="en"):
    """convert a example into necessary features"""
    # Get the label
    label = example["label"]
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    # Convert raw text to feature
    example = tokenizer(premise,
                        text_pair=hypothesis,
                        max_length=max_seq_length,
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        lang=language)
    return example["input_ids"], example["attention_mask"], label


def get_test_dataloader(args, language, tokenizer):
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=0, dtype="int64"),  # attention_mask
        Stack(dtype="int64")  # labels
    ): fn(samples)
    # make sure language is `language``
    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         language=language)
    test_ds = load_dataset("xnli", language, splits="test")
    test_ds = test_ds.map(trans_func, lazy=True)
    test_batch_sampler = BatchSampler(test_ds,
                                      batch_size=args.batch_size * 4,
                                      shuffle=False)
    test_data_loader = DataLoader(dataset=test_ds,
                                  batch_sampler=test_batch_sampler,
                                  collate_fn=batchify_fn,
                                  num_workers=0,
                                  return_list=True)
    return test_data_loader


def do_eval(args):
    paddle.set_device(args.device)
    tokenizer = XLMTokenizer.from_pretrained(args.model_name_or_path)
    model = XLMForSequenceClassification.from_pretrained(
        args.model_name_or_path)
    model.eval()
    metric = Accuracy()
    all_languages_acc = []
    for language in all_languages:
        test_dataloader = get_test_dataloader(args, language, tokenizer)
        acc = evaluate(model, metric, test_dataloader, language, tokenizer)
        all_languages_acc.append(acc)
    print("test mean acc: %.4f" % np.mean(all_languages_acc))


def print_arguments(args):
    """print arguments"""
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_eval(args)
