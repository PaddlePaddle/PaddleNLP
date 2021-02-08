# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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
"""Network for auto dialogue evaluation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid


def create_net(is_training,
               model_input,
               args,
               clip_value=10.0,
               word_emb_name="shared_word_emb",
               lstm_W_name="shared_lstm_W",
               lstm_bias_name="shared_lstm_bias"):

    context_wordseq = model_input.context_wordseq
    response_wordseq = model_input.response_wordseq
    label = model_input.labels

    #emb
    context_emb = fluid.embedding(
        input=context_wordseq,
        size=[args.vocab_size, args.emb_size],
        is_sparse=True,
        param_attr=fluid.ParamAttr(
            name=word_emb_name,
            initializer=fluid.initializer.Normal(scale=0.1)))

    response_emb = fluid.embedding(
        input=response_wordseq,
        size=[args.vocab_size, args.emb_size],
        is_sparse=True,
        param_attr=fluid.ParamAttr(
            name=word_emb_name,
            initializer=fluid.initializer.Normal(scale=0.1)))

    #fc to fit dynamic LSTM
    context_fc = fluid.layers.fc(input=context_emb,
                                 size=args.hidden_size * 4,
                                 param_attr=fluid.ParamAttr(name='fc_weight'),
                                 bias_attr=fluid.ParamAttr(name='fc_bias'))

    response_fc = fluid.layers.fc(input=response_emb,
                                  size=args.hidden_size * 4,
                                  param_attr=fluid.ParamAttr(name='fc_weight'),
                                  bias_attr=fluid.ParamAttr(name='fc_bias'))

    #LSTM
    context_rep, _ = fluid.layers.dynamic_lstm(
        input=context_fc,
        size=args.hidden_size * 4,
        param_attr=fluid.ParamAttr(name=lstm_W_name),
        bias_attr=fluid.ParamAttr(name=lstm_bias_name))
    context_rep = fluid.layers.sequence_last_step(context_rep)

    response_rep, _ = fluid.layers.dynamic_lstm(
        input=response_fc,
        size=args.hidden_size * 4,
        param_attr=fluid.ParamAttr(name=lstm_W_name),
        bias_attr=fluid.ParamAttr(name=lstm_bias_name))
    response_rep = fluid.layers.sequence_last_step(input=response_rep)

    logits = fluid.layers.bilinear_tensor_product(
        context_rep, response_rep, size=1)

    if args.loss_type == 'CLS':
        label = fluid.layers.cast(x=label, dtype='float32')
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
        loss = fluid.layers.reduce_mean(
            fluid.layers.clip(
                loss, min=-clip_value, max=clip_value))
    elif args.loss_type == 'L2':
        norm_score = 2 * fluid.layers.sigmoid(logits)
        label = fluid.layers.cast(x=label, dtype='float32')
        loss = fluid.layers.square_error_cost(norm_score, label) / 4
        loss = fluid.layers.reduce_mean(loss)
    else:
        raise ValueError

    if is_training:
        return loss
    else:
        return logits


def set_word_embedding(word_emb, place, word_emb_name="shared_word_emb"):
    """
    Set word embedding
    """
    word_emb_param = fluid.global_scope().find_var(word_emb_name).get_tensor()
    word_emb_param.set(word_emb, place)
