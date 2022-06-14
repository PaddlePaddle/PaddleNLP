# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import h5py
import numpy as np

import paddle
from paddle.io import DataLoader, Dataset
from paddlenlp.data import Tuple, Stack


def create_pretraining_dataset(input_file,
                               max_pred_length,
                               args,
                               data_holders,
                               worker_init=None,
                               places=None):
    train_data = PretrainingDataset(input_file=input_file,
                                    max_pred_length=max_pred_length)
    train_batch_sampler = paddle.io.BatchSampler(train_data,
                                                 batch_size=args.batch_size,
                                                 shuffle=True)

    def _collate_data(data, stack_fn=Stack()):
        num_fields = len(data[0])
        out = [None] * num_fields
        # input_ids, segment_ids, input_mask, masked_lm_positions,
        # masked_lm_labels, next_sentence_labels, mask_token_num
        for i in (0, 1, 2, 5):
            out[i] = stack_fn([x[i] for x in data])
        batch_size, seq_length = out[0].shape
        size = num_mask = sum(len(x[3]) for x in data)
        # Padding for divisibility by 8 for fp16 or int8 usage
        if size % 8 != 0:
            size += 8 - (size % 8)
        # masked_lm_positions
        # Organize as a 1D tensor for gather or use gather_nd
        out[3] = np.full(size, 0, dtype=np.int32)
        # masked_lm_labels
        out[4] = np.full([size, 1], -1, dtype=np.int64)
        mask_token_num = 0
        for i, x in enumerate(data):
            for j, pos in enumerate(x[3]):
                out[3][mask_token_num] = i * seq_length + pos
                out[4][mask_token_num] = x[4][j]
                mask_token_num += 1
        # mask_token_num
        out.append(np.asarray([mask_token_num], dtype=np.float32))
        if args.use_amp and args.use_pure_fp16:
            # cast input_mask to fp16
            out[2] = out[2].astype(np.float16)
            # cast masked_lm_scale to fp16
            out[-1] = out[-1].astype(np.float16)
        return out

    train_data_loader = DataLoader(dataset=train_data,
                                   places=places,
                                   feed_list=data_holders,
                                   batch_sampler=train_batch_sampler,
                                   collate_fn=_collate_data,
                                   num_workers=0,
                                   worker_init_fn=worker_init,
                                   return_list=False)
    return train_data_loader, input_file


def create_data_holder(args):
    input_ids = paddle.static.data(name="input_ids",
                                   shape=[-1, -1],
                                   dtype="int64")
    segment_ids = paddle.static.data(name="segment_ids",
                                     shape=[-1, -1],
                                     dtype="int64")
    input_mask = paddle.static.data(name="input_mask",
                                    shape=[-1, 1, 1, -1],
                                    dtype="float32")
    masked_lm_positions = paddle.static.data(name="masked_lm_positions",
                                             shape=[-1],
                                             dtype="int32")
    masked_lm_labels = paddle.static.data(name="masked_lm_labels",
                                          shape=[-1, 1],
                                          dtype="int64")
    next_sentence_labels = paddle.static.data(name="next_sentence_labels",
                                              shape=[-1, 1],
                                              dtype="int64")
    masked_lm_scale = paddle.static.data(name="masked_lm_scale",
                                         shape=[-1, 1],
                                         dtype="float32")
    return [
        input_ids, segment_ids, input_mask, masked_lm_positions,
        masked_lm_labels, next_sentence_labels, masked_lm_scale
    ]


class PretrainingDataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            'input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions',
            'masked_lm_ids', 'next_sentence_labels'
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [
            input_ids, input_mask, segment_ids, masked_lm_positions,
            masked_lm_ids, next_sentence_labels
        ] = [
            input[index].astype(np.int64)
            if indice < 5 else np.asarray(input[index].astype(np.int64))
            for indice, input in enumerate(self.inputs)
        ]
        # TODO: whether to use reversed mask by changing 1s and 0s to be
        # consistent with nv bert
        input_mask = (1 - np.reshape(input_mask.astype(np.float32),
                                     [1, 1, input_mask.shape[0]])) * -1e4

        index = self.max_pred_length
        # store number of  masked tokens in index
        # outputs of torch.nonzero diff with that of numpy.nonzero by zip
        padded_mask_indices = (masked_lm_positions == 0).nonzero()[0]
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
            mask_token_num = index
        else:
            index = self.max_pred_length
            mask_token_num = self.max_pred_length
        # masked_lm_labels = np.full(input_ids.shape, -1, dtype=np.int64)
        # masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        masked_lm_labels = masked_lm_ids[:index]
        masked_lm_positions = masked_lm_positions[:index]
        # softmax_with_cross_entropy enforce last dim size equal 1
        masked_lm_labels = np.expand_dims(masked_lm_labels, axis=-1)
        next_sentence_labels = np.expand_dims(next_sentence_labels, axis=-1)

        return [
            input_ids, segment_ids, input_mask, masked_lm_positions,
            masked_lm_labels, next_sentence_labels
        ]
