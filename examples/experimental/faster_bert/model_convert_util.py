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
import logging
import os
import sys
import random
import time
import math
import distutils.util
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from static.modeling import FasterBertForSequenceClassification
from paddlenlp.transformers import ElectraForSequenceClassification, ElectraTokenizer
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "mrpc": AccuracyAndF1,
    "sts-b": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
}

MODEL_CLASSES = {
    "bert": (BertForSequenceClassification, BertTokenizer),
    "faster_bert": (FasterBertForSequenceClassification, BertTokenizer),
    "ernie": (ErnieForSequenceClassification, ErnieTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu", "npu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu/npu."
    )
    parser.add_argument(
        "--use_amp",
        type=distutils.util.strtobool,
        default=False,
        help="Enable mixed precision training.")
    args = parser.parse_args()
    return args

def fused_weight(weight, num_head):
    if paddle.in_dynamic_mode():
        a = paddle.transpose(weight, perm=[1, 0])
        return paddle.reshape(a, shape=[1, num_head, int(a.shape[0]/num_head), a.shape[1]])
    else:
        a = weight.transpose(1, 0)
        return a.reshape((1, num_head, int(a.shape[0]/num_head), a.shape[1]))

def fused_qkv(q, k, v, num_head):
    fq = fused_weight(q, num_head)
    fk = fused_weight(k, num_head)
    fv = fused_weight(v, num_head)
    if paddle.in_dynamic_mode():
        return paddle.concat(x=[fq, fk, fv], axis=0)
    else:
        return np.concatenate((fq, fk, fv), axis=0)

def convert_encoder(encoder, fused_encoder, num_heads):
    for i in range(len(encoder.layers)):
        base_layer = encoder.layers[i]
        fused_layer = fused_encoder.layers[i]
        fused_layer.ffn._linear1_weight.set_value(base_layer.linear1.weight)
        fused_layer.ffn._linear1_bias.set_value(base_layer.linear1.bias)
        fused_layer.ffn._linear2_weight.set_value(base_layer.linear2.weight)
        fused_layer.ffn._linear2_bias.set_value(base_layer.linear2.bias)
        fused_layer.ffn._ln1_scale.set_value(base_layer.norm2.weight)
        fused_layer.ffn._ln1_bias.set_value(base_layer.norm2.bias)
        fused_layer.ffn._ln2_scale.set_value(base_layer.norm2.weight)
        fused_layer.ffn._ln2_bias.set_value(base_layer.norm2.bias)

        fused_layer.fused_attn.linear_weight.set_value(base_layer.self_attn.out_proj.weight)
        fused_layer.fused_attn.linear_bias.set_value(base_layer.self_attn.out_proj.bias)
        fused_layer.fused_attn.pre_ln_scale.set_value(base_layer.norm1.weight)
        fused_layer.fused_attn.pre_ln_bias.set_value(base_layer.norm1.bias)
        fused_layer.fused_attn.ln_scale.set_value(base_layer.norm1.weight)
        fused_layer.fused_attn.ln_bias.set_value(base_layer.norm1.bias)

        q = base_layer.self_attn.q_proj.weight
        q_bias = base_layer.self_attn.q_proj.bias
        k = base_layer.self_attn.k_proj.weight
        k_bias = base_layer.self_attn.k_proj.bias
        v = base_layer.self_attn.v_proj.weight
        v_bias = base_layer.self_attn.v_proj.bias

        qkv_weight = fused_qkv(q, k, v, num_heads)
        fused_layer.fused_attn.qkv_weight.set_value(qkv_weight)

        if paddle.in_dynamic_mode():
            tmp = paddle.concat(x=[q_bias, k_bias, v_bias], axis=0)
            qkv_bias = paddle.reshape(tmp, shape=[3, num_heads, int(tmp.shape[0]/3/num_heads)])
            fused_layer.fused_attn.qkv_bias.set_value(qkv_bias)
        else:
            qkv_bias = np.concatenate((q, k, v), axis=0)
            fused_layer.fused_attn.qkv_bias.set_value(qkv_bias)

def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)

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
        label = example['labels']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if (int(is_test) + len(example)) == 2:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    else:
        example = tokenizer(
            example['sentence1'],
            text_pair=example['sentence2'],
            max_seq_len=max_seq_length)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']

def do_convert(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    faster_model_class, faster_tokenizer_class = MODEL_CLASSES["faster_" + args.model_type]

    train_ds = load_dataset('chnsenticorp', splits="train")
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=args.max_seq_length)
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)

    num_classes = 1 if train_ds.label_list == None else len(train_ds.label_list)
    base_model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)
    fused_model = faster_model_class.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)

    num_heads = fused_model.bert.encoder.layers[0].fused_attn.num_heads
    convert_encoder(base_model.bert.encoder, fused_model.bert.encoder, num_heads)

    model = fused_model

    output_dir = os.path.join(args.output_dir,
                              "faster_%s_%s" %
                              (args.model_type, args.task_name))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model._layers if isinstance(
        model, paddle.DataParallel) else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    args = parse_args()
    #print_arguments(args)
    do_convert(args)
