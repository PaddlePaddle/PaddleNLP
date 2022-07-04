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
"""padding and batching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def pad_batch_data(insts,
                   pretraining_task='seq2seq',
                   pad_idx=1,
                   sent_b_starts=None,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype('int64').reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype('int64').reshape([-1, max_len, 1])]

    if return_input_mask:
        if pretraining_task is 'seq2seq':
            assert sent_b_starts is not None, \
                "[FATAL] For seq2seq lanugae model loss," \
                " sent_b_starts should not be None"
            # This is used to avoid attention on paddings and subsequent words.
            input_mask_data = np.zeros((inst_data.shape[0], max_len, max_len))
            for index, mask_data in enumerate(input_mask_data):
                start = sent_b_starts[index]
                end = len(insts[index])
                mask_data[:end, :start] = 1.0
                # Generate the lower triangular matrix using the slice of matrix
                b = np.tril(np.ones([end - start, end - start]), 0)
                mask_data[start:end, start:end] = b
            input_mask_data = np.array(input_mask_data).reshape([-1, max_len, max_len])
        else:
            # This is used to avoid attention on paddings.
            input_mask_data = np.array([[1] * len(inst) + [0] *
                                        (max_len - len(inst)) for inst in insts])
            input_mask_data = np.expand_dims(input_mask_data, axis=-1)
            # input_mask_data = np.matmul(input_mask_data, np.transpose(input_mask_data, (0, 2, 1)))
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype('int64').reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


def pad_feature_data(data, pad_value=0.0, dtype="float32", return_mask=False, batch_image_size=None):
    """for image feature sequence padding"""
    # num box + 1 ,1 for global feature
    max_lenth = max([len(item) for item in data])
    data_width = len(data[0][0])
    out_data = np.ones((len(data), max_lenth, data_width), dtype=dtype) * pad_value
    out_mask = np.zeros((len(data), max_lenth, 1), dtype=dtype)
    for i in range(len(data)):
        out_data[i, 0:len(data[i]), :] = data[i]
        if return_mask and batch_image_size[i] > 1:
            out_mask[i, 0:len(data[i]), :] = 1.0
    if return_mask:
        return out_data, out_mask
    else:
        return out_data


def gen_seq2seq_mask(insts, sent_b_starts=None):
    """
    generate input mask for seq2seq
    """
    max_len = max(len(inst) for inst in insts)
    input_mask_data = np.zeros((len(insts), max_len, max_len))
    for index, mask_data in enumerate(input_mask_data):
        start = sent_b_starts[index]
        end = len(insts[index])
        mask_data[:end, :start] = 1.0
        # Generate the lower triangular matrix using the slice of matrix
        b = np.tril(np.ones([end - start, end - start]), 0)
        mask_data[start:end, start:end] = b
    input_mask_data = np.array(input_mask_data, dtype='float32').reshape([-1, max_len, max_len])
    return input_mask_data


if __name__ == "__main__":
    pass
