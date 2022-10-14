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

import os
import random
import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.optimizer.lr import LambdaDecay
from paddlenlp.datasets import MapDataset

from data import InputFeatures


def set_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def check_args(args):
    """check output_dir and make it when not exist"""
    if os.path.exists(args.output_dir):
        if os.listdir(args.output_dir) and not args.overwrite_output:
            raise ValueError('Path Configuration: output_dir {} exists!'.format(
                args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.dataset = args.dataset.lower()


def convert_example(example, template, tokenizer_wrapper, verbalizer=None):
    if verbalizer is not None and hasattr(verbalizer, 'wrap_one_example'):
        exmaple = verbalizer.wrap_one_example(example)
    example = template.wrap_one_example(example)
    encoded_inputs = InputFeatures(
        **tokenizer_wrapper.tokenize_one_example(example), **example[1])
    return encoded_inputs


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if isinstance(dataset, list):
        dataset = MapDataset(dataset)
    assert isinstance(dataset, MapDataset)

    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)

    return DataLoader(dataset=dataset,
                      batch_sampler=batch_sampler,
                      collate_fn=batchify_fn,
                      return_list=True)


class LinearSchedulerWarmup(LambdaDecay):
    """
    Linear scheduler with warm up.
    """

    def __init__(self,
                 learning_rate,
                 warmup_steps,
                 max_steps,
                 last_epoch=-1,
                 verbose=False):

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(max_steps - current_step) /
                float(max(1, max_steps - warmup_steps)))

        super().__init__(learning_rate, lr_lambda, last_epoch, verbose)
