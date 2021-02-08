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

import paddle.fluid as fluid

from model.bert import BertModel


def create_model(args, bert_config, num_labels, is_prediction=False):
    input_fields = {
        'names': ['src_ids', 'pos_ids', 'sent_ids', 'input_mask', 'labels'],
        'shapes': [[None, None], [None, None], [None, None], [None, None, 1],
                   [None, 1]],
        'dtypes': ['int64', 'int64', 'int64', 'float32', 'int64'],
        'lod_levels': [0, 0, 0, 0, 0],
    }

    inputs = [
        fluid.data(
            name=input_fields['names'][i],
            shape=input_fields['shapes'][i],
            dtype=input_fields['dtypes'][i],
            lod_level=input_fields['lod_levels'][i])
        for i in range(len(input_fields['names']))
    ]
    (src_ids, pos_ids, sent_ids, input_mask, labels) = inputs

    data_loader = fluid.io.DataLoader.from_generator(
        feed_list=inputs, capacity=50, iterable=False)

    bert = BertModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=bert_config,
        use_fp16=args.use_fp16)

    cls_feats = bert.get_pooled_output()
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        input=cls_feats,
        num_flatten_dims=2,
        size=num_labels,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

    if is_prediction:
        probs = fluid.layers.softmax(logits)
        feed_targets_name = [
            src_ids.name, pos_ids.name, sent_ids.name, input_mask.name
        ]
        return data_loader, probs, feed_targets_name

    logits = fluid.layers.reshape(logits, [-1, num_labels], inplace=True)
    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)

    return data_loader, loss, probs, accuracy, num_seqs
