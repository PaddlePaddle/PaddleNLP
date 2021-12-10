#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import six
import sys
import os
import time
import argparse
from functools import partial
import numpy as np

import paddle
from paddle.metric import Accuracy

import paddlenlp
import paddleslim
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    "--task_name", type=str, default="afqmc", required=False, help="task_name")
parser.add_argument(
    "--input_dir",
    type=str,
    default="afqmc",
    required=False,
    help="input task model dire")

args = parser.parse_args()

METRIC_CLASSES = {
    "afqmc": Accuracy,
    "tnews": Accuracy,
    "iflytek": Accuracy,
    "ocnli": Accuracy,
    "cmnli": Accuracy,
    "cluewsc2020": Accuracy,
    "csl": Accuracy,
}

MODEL_CLASSES = {"ernie": (ErnieForSequenceClassification, ErnieTokenizer), }


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['label']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if 'sentence' in example:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    elif 'sentence1' in example:
        example = tokenizer(
            example['sentence1'],
            text_pair=example['sentence2'],
            max_seq_len=max_seq_length)
    elif 'keyword' in example:  # CSL
        sentence1 = " ".join(example['keyword'])
        example = tokenizer(
            sentence1, text_pair=example['abst'], max_seq_len=max_seq_length)
    elif 'target' in example:  # wsc
        text, query, pronoun, query_idx, pronoun_idx = example['text'], example[
            'target']['span1_text'], example['target']['span2_text'], example[
                'target']['span1_index'], example['target']['span2_index']
        text_list = list(text)
        assert text[pronoun_idx:(pronoun_idx + len(pronoun)
                                 )] == pronoun, "pronoun: {}".format(pronoun)
        assert text[query_idx:(query_idx + len(query)
                               )] == query, "query: {}".format(query)
        if pronoun_idx > query_idx:
            text_list.insert(query_idx, "_")
            text_list.insert(query_idx + len(query) + 1, "_")
            text_list.insert(pronoun_idx + 2, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
        else:
            text_list.insert(pronoun_idx, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 1, "]")
            text_list.insert(query_idx + 2, "_")
            text_list.insert(query_idx + len(query) + 2 + 1, "_")
        text = "".join(text_list)
        example = tokenizer(text, max_seq_len=max_seq_length)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']


def quant_post(args, batch_size=8, algo='avg'):
    paddle.enable_static()
    place = paddle.set_device("gpu")
    exe = paddle.static.Executor(place)
    args.task_name = args.task_name.lower()

    train_ds = paddlenlp.datasets.load_dataset(
        "clue", args.task_name, splits="dev")

    tokenizer = ErnieTokenizer.from_pretrained(
        "../ernie-batchbatch-50w_400000/best_models/AFQMC/")

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=128,
        is_test=True)
    train_ds = train_ds.map(trans_func, lazy=True)

    def test():
        batch_data = [[], []]
        for data in train_ds:
            batch_data[0].append(data[0])
            batch_data[1].append(data[1])
            if len(batch_data[0]) == batch_size:
                input_ids = Pad(axis=0, pad_val=0)(batch_data[0])
                segment_ids = Pad(axis=0, pad_val=0)(batch_data[1])
                ones = np.ones_like(input_ids, dtype="int64")
                seq_length = np.cumsum(ones, axis=-1)

                position_ids = seq_length - ones
                attention_mask = np.expand_dims(
                    (input_ids == 0).astype("float32") * -1e9, axis=[1, 2])
                yield [input_ids, segment_ids]
                batch_data = [[], []]

    paddleslim.quant.quant_post_static(
        exe,
        args.input_dir,
        os.path.join(args.task_name + '_quant_models', algo + str(batch_size)),
        save_model_filename='int8.pdmodel',
        save_params_filename='int8.pdiparams',
        algo=algo,
        hist_percent=0.9999,
        batch_generator=test,
        model_filename='float.pdmodel',
        params_filename='float.pdiparams',
        quantizable_op_type=['matmul', 'matmul_v2'],
        weight_bits=8,
        weight_quantize_type='channel_wise_abs_max',
        batch_nums=1, )


if __name__ == '__main__':
    paddle.enable_static()
    for batch_size in [4, 8]:
        for algo in ['abs_max', 'avg', 'mse', 'hist']:
            quant_post(args, batch_size, algo)
