#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import paddle.fluid as fluid

import modeling
from model.xlnet import XLNetModel, _get_initializer


def get_regression_loss(args, xlnet_config, features, is_training=False):
    """Loss for downstream regression tasks."""

    inp = fluid.layers.transpose(features["input_ids"], [1, 0, 2])
    seg_id = features["segment_ids"]
    inp_mask = fluid.layers.transpose(features["input_mask"], [1, 0])
    label = features["label_ids"]

    xlnet_model = XLNetModel(
        input_ids=inp,
        seg_ids=seg_id,
        input_mask=inp_mask,
        xlnet_config=xlnet_config,
        args=args)

    summary = xlnet_model.get_pooled_out(args.summary_type, args.use_summ_proj)
    per_example_loss, logits = modeling.regression_loss(
        hidden=summary,
        labels=label,
        initializer=_get_initializer(args),
        name="model_regression_{}".format(args.task_name.lower()),
        return_logits=True)

    total_loss = fluid.layers.reduce_mean(per_example_loss)

    return total_loss, per_example_loss, logits


def get_classification_loss(args,
                            xlnet_config,
                            features,
                            n_class,
                            is_training=True):
    """Loss for downstream classification tasks."""

    inp = fluid.layers.transpose(features["input_ids"], [1, 0, 2])
    seg_id = features["segment_ids"]
    inp_mask = fluid.layers.transpose(features["input_mask"], [1, 0])
    label = features["label_ids"]

    xlnet_model = XLNetModel(
        input_ids=inp,
        seg_ids=seg_id,
        input_mask=inp_mask,
        xlnet_config=xlnet_config,
        args=args)

    summary = xlnet_model.get_pooled_out(args.summary_type, args.use_summ_proj)

    per_example_loss, logits = modeling.classification_loss(
        hidden=summary,
        labels=label,
        n_class=n_class,
        initializer=xlnet_model.get_initializer(),
        name="model_classification_{}".format(args.task_name),
        return_logits=True)

    total_loss = fluid.layers.reduce_mean(per_example_loss)
    return total_loss, per_example_loss, logits


def create_model(args, xlnet_config, n_class, is_training=False):
    label_ids_type = 'int64' if n_class else 'float32'
    input_fields = {
        'names': [
            'input_ids', 'input_mask', 'segment_ids', 'label_ids',
            'is_real_example'
        ],
        'shapes': [[-1, args.max_seq_length, 1], [-1, args.max_seq_length],
                   [-1, args.max_seq_length], [-1, 1], [-1, 1]],
        'dtypes':
        ['int64', 'float32', 'int64', 'int64', label_ids_type, 'int64'],
        'lod_levels': [0, 0, 0, 0, 0, 0],
    }

    inputs = [
        fluid.layers.data(
            name=input_fields['names'][i],
            shape=input_fields['shapes'][i],
            dtype=input_fields['dtypes'][i],
            lod_level=input_fields['lod_levels'][i])
        for i in range(len(input_fields['names']))
    ]
    (input_ids, input_mask, segment_ids, label_ids, is_real_example) = inputs

    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=inputs, capacity=50, iterable=False)

    features = collections.OrderedDict()
    features["input_ids"] = input_ids
    features["input_mask"] = input_mask
    features["segment_ids"] = segment_ids
    features["label_ids"] = label_ids
    features["is_real_example"] = is_real_example

    if args.is_regression:
        (total_loss, per_example_loss, logits) = get_regression_loss(
            args, xlnet_config, features, is_training)
    else:
        (total_loss, per_example_loss, logits) = get_classification_loss(
            args, xlnet_config, features, n_class, is_training)

    num_seqs = fluid.layers.fill_constant_batch_size_like(
        input=label_ids, shape=[-1, 1], value=1, dtype="int64")
    num_seqs = fluid.layers.reduce_sum(num_seqs)

    return data_loader, total_loss, logits, num_seqs, label_ids
