# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def cal_pt_loss(teacher_hidden_states, student_hidden_states, strategy, k):
    """Only support two strategies: skip and last"""
    if strategy == "skip":
        index_func = lambda student_idx: (student_idx + 1) * k - 1
    else:
        index_func = lambda student_idx: student_idx - k + 1

    mse_loss = nn.MSELoss()
    pt_loss = 0
    # TODO(liujiaqi06): Check k and layers of two models.
    for student_layer_idx in range(len(student_hidden_states)):
        teacher_layer_idx = index_func(student_layer_idx)
        student_hidden_normalized = F.normalize(student_hidden_states[
            student_layer_idx])
        teacher_hidden_normalized = F.normalize(teacher_hidden_states[
            teacher_layer_idx])
        mse = mse_loss(student_hidden_normalized, teacher_hidden_normalized)
        pt_loss += mse
    return pt_loss


def cal_pkd_loss(args,
                 student_logits,
                 labels,
                 teacher_logits=None,
                 teacher_hidden_states=None,
                 student_hidden_states=None):
    # Cross entropy loss calculates between student_logits and labels
    # import pdb; pdb.set_trace()
    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
    if teacher_logits is None:
        return ce_loss

    teacher_logits = teacher_logits / args.T
    teacher_logits = F.softmax(teacher_logits)
    # Distance between teacher's prediction and student's prediction
    ds_loss = nn.CrossEntropyLoss(soft_label=True)(student_logits,
                                                   teacher_logits)
    # print(ce_loss, ds_loss)
    if teacher_hidden_states is None:
        return (1 - args.alpha) * ce_loss + args.alpha * ds_loss
    # Intermediate layer's loss
    pt_loss = cal_pt_loss(teacher_hidden_states, student_hidden_states,
                          args.strategy, args.k)
    # import pdb; pdb.set_trace()
    print(ce_loss.numpy()[0], ds_loss.numpy()[0], pt_loss.numpy()[0])
    loss = (1 - args.alpha
            ) * ce_loss + args.alpha * ds_loss + args.beta * pt_loss
    return loss, ce_loss, ds_loss, pt_loss
