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

from copy import deepcopy

import numpy as np


def convert_example(tokenizer,
                    attn_id,
                    tgt_type_id=3,
                    max_encode_len=512,
                    max_decode_len=128,
                    is_test=False,
                    noise_prob=0.,
                    use_random_noice=False):

    def warpper(example):
        """convert an example into necessary features"""
        tokens = example['tokens']
        labels = example['labels']
        encoded_src = tokenizer(tokens,
                                max_seq_len=max_encode_len,
                                pad_to_max_seq_len=False)
        src_ids, src_sids = encoded_src["input_ids"], encoded_src[
            "token_type_ids"]
        src_pids = np.arange(len(src_ids))

        if not is_test:
            encoded_tgt = tokenizer(labels,
                                    max_seq_len=max_decode_len,
                                    pad_to_max_seq_len=False)
            tgt_ids, tgt_sids = encoded_tgt["input_ids"], encoded_tgt[
                "token_type_ids"]
            tgt_ids = np.array(tgt_ids).astype("int64")
            tgt_sids = np.array(tgt_sids) + tgt_type_id
            tgt_pids = np.arange(len(tgt_ids)) + len(src_ids)

        attn_ids = np.ones_like(tgt_ids) * attn_id
        if noise_prob > 0.:
            tgt_labels = deepcopy(tgt_ids)
            if use_random_noice:
                noice_ids = np.random.randint(1,
                                              len(tokenizer.vocab),
                                              size=tgt_ids.shape)
            else:
                noice_ids = np.ones_like(tgt_ids) * tokenizer.vocab['[NOISE]']
            pos, = np.where(np.ones_like(tgt_ids))
            np.random.shuffle(pos)
            pos = pos[:int(noise_prob * len(pos))]
            tgt_ids[pos, ] = noice_ids[pos, ]
        else:
            tgt_labels = tgt_ids

        return (src_ids, src_pids, src_sids, tgt_ids, tgt_pids, tgt_sids,
                attn_ids, tgt_labels)

    return warpper


def gen_mask(batch_ids, mask_type='bidi', query_len=None, pad_value=0):
    if query_len is None:
        query_len = batch_ids.shape[1]
    if mask_type != 'empty':
        mask = (batch_ids != pad_value).astype(np.float32)
        mask = np.tile(np.expand_dims(mask, 1), [1, query_len, 1])
        if mask_type == 'causal':
            assert query_len == batch_ids.shape[1]
            mask = np.tril(mask)
        elif mask_type == 'causal_without_diag':
            assert query_len == batch_ids.shape[1]
            mask = np.tril(mask, -1)
        elif mask_type == 'diag':
            assert query_len == batch_ids.shape[1]
            # import pdb; pdb.set_trace()
            mask = np.stack([np.diag(np.diag(m)) for m in mask], 0)

    else:
        mask_type == 'empty'
        mask = np.zeros_like(batch_ids).astype(np.float32)
        mask = np.tile(np.expand_dims(mask, 1), [1, query_len, 1])
    return mask


def after_padding(args):
    '''
    attention mask:
    ***  src,  tgt, attn
    src  00,   01,   11
    tgt  10,   11,   12
    attn 20,   21,   22

    ***   s1, s2 | t1 t2 t3| attn1 attn2 attn3
    s1    1,  1  | 0, 0, 0,| 0,    0,    0,
    s2    1,  1  | 0, 0, 0,| 0,    0,    0,
    -
    t1    1,  1, | 1, 0, 0,| 0,    0,    0,
    t2    1,  1, | 1, 1, 0,| 0,    0,    0,
    t3    1,  1, | 1, 1, 1,| 0,    0,    0,
    -
    attn1 1,  1, | 0, 0, 0,| 1,    0,    0,
    attn2 1,  1, | 1, 0, 0,| 0,    1,    0,
    attn3 1,  1, | 1, 1, 0,| 0,    0,    1,

    for details, see Fig3. https://arxiv.org/abs/2001.11314
    '''
    src_ids, src_pids, src_sids, tgt_ids, tgt_pids, tgt_sids, attn_ids, tgt_labels = args
    src_len = src_ids.shape[1]
    tgt_len = tgt_ids.shape[1]
    mask_00 = gen_mask(src_ids, 'bidi', query_len=src_len)
    mask_01 = gen_mask(tgt_ids, 'empty', query_len=src_len)
    mask_02 = gen_mask(attn_ids, 'empty', query_len=src_len)

    mask_10 = gen_mask(src_ids, 'bidi', query_len=tgt_len)
    mask_11 = gen_mask(tgt_ids, 'causal', query_len=tgt_len)
    mask_12 = gen_mask(attn_ids, 'empty', query_len=tgt_len)

    mask_20 = gen_mask(src_ids, 'bidi', query_len=tgt_len)
    mask_21 = gen_mask(tgt_ids, 'causal_without_diag', query_len=tgt_len)
    mask_22 = gen_mask(attn_ids, 'diag', query_len=tgt_len)

    mask_src_2_src = mask_00
    mask_tgt_2_srctgt = np.concatenate([mask_10, mask_11], 2)
    mask_attn_2_srctgtattn = np.concatenate([mask_20, mask_21, mask_22], 2)

    raw_tgt_labels = deepcopy(tgt_labels)
    tgt_labels = tgt_labels[np.where(tgt_labels != 0)]
    return (src_ids, src_sids, src_pids, tgt_ids, tgt_sids, tgt_pids, attn_ids,
            mask_src_2_src, mask_tgt_2_srctgt, mask_attn_2_srctgtattn,
            tgt_labels, raw_tgt_labels)
