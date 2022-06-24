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


import paddle

import os
import random
import numpy as np


def get_special_tokens():
    return ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]

def get_special_token_ids(tokenizer):
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    return tokenizer.convert_tokens_to_ids(special_tokens)

def get_special_token_dict(tokenizer):
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    special_token_dict = dict(zip(special_tokens, tokenizer.convert_tokens_to_ids(special_tokens)))
    return special_token_dict


def create_dataloader(dataset, mode="train", batch_size=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)
    shuffle = True if mode == "train" else False
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)


def convert_example(example, tokenizer, max_seq_length=512, do_evalute=False):
    result = []
    for key, text in example.items():
        if "label" in key:
            # do_evaluate
            result += [example["label"]]
        else:
            # do_train
            encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length, return_attention_mask=True)
            # print(encoded_inputs)
            input_ids = encoded_inputs["input_ids"]
            token_type_ids = encoded_inputs["token_type_ids"]
            attention_mask = encoded_inputs["attention_mask"]
            result += [input_ids, token_type_ids, attention_mask]
    return result


def read_text_single(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = line.rstrip()
            yield {"text_a": data, "text_b": data}

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

def mask_tokens(batch_inputs, tokenizer, mlm_probability=0.15):
    """
    Description: Mask input_ids for masked language modeling: 80% MASK, 10% random, 10% original
    """
    mlm_inputs = batch_inputs.clone()
    mlm_labels = batch_inputs.clone()

    # print(mlm_inputs)
    probability_matrix = paddle.full(mlm_inputs.shape, mlm_probability)
    # print(probability_matrix)
    
    special_tokens_mask = paddle.cast(paddle.zeros(mlm_inputs.shape), dtype=bool)
    for special_token_id in get_special_token_ids(tokenizer):
        special_tokens_mask |= (mlm_inputs==special_token_id)

    
    probability_matrix = masked_fill(probability_matrix, special_tokens_mask, 0.0)
    # print(probability_matrix)

    masked_indices = paddle.cast(paddle.bernoulli(probability_matrix), dtype=bool)
    mlm_labels = masked_fill(mlm_labels, ~masked_indices, -100)

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = paddle.cast(paddle.bernoulli(paddle.full(mlm_inputs.shape, 0.8)), dtype=bool) & masked_indices
    mlm_inputs = masked_fill(mlm_inputs, indices_replaced, tokenizer.mask_token_id)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = paddle.cast(paddle.bernoulli(paddle.full(mlm_inputs.shape, 0.5)) ,dtype=bool) & masked_indices & ~indices_replaced
    random_words = paddle.randint(0, len(tokenizer), mlm_inputs.shape, dtype=mlm_inputs.dtype)
    mlm_inputs = paddle.where(indices_random, random_words, mlm_inputs)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    # print(mlm_inputs)
    # print(mlm_labels)
    return mlm_inputs, mlm_labels


def read_text_pair(data_path, is_infer=False):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_infer:
                if len(data[0]) == 0 or len(data[1]) == 0:
                    continue
                yield {"text_a": data[0], "text_b": data[1]}
            else:
                if len(data[0]) == 0 or len(data[1]) == 0 or len(data[2]) == 0:
                    continue
                yield {"text_a": data[0], "text_b": data[1], "label": data[2]}


def word_repetition(input_ids, token_type_ids, dup_rate=0.32):
    """Word Reptition Strategy"""
    input_ids = input_ids.numpy().tolist()
    token_type_ids = token_type_ids.numpy().tolist()

    batch_size, seq_len = len(input_ids), len(input_ids[0])
    repetitied_input_ids = []
    repetitied_token_type_ids = []
    rep_seq_len = seq_len
    for batch_id in range(batch_size):
        cur_input_id = input_ids[batch_id]
        actual_len = np.count_nonzero(cur_input_id)
        dup_word_index = []
        # If sequence length is less than 5, skip it
        if (actual_len > 5):
            dup_len = random.randint(a=0, b=max(2, int(dup_rate * actual_len)))
            # Skip cls and sep position
            dup_word_index = random.sample(list(range(1, actual_len - 1)), k=dup_len)

        r_input_id = []
        r_token_type_id = []
        for idx, word_id in enumerate(cur_input_id):
            # Insert duplicate word
            if idx in dup_word_index:
                r_input_id.append(word_id)
                r_token_type_id.append(token_type_ids[batch_id][idx])
            r_input_id.append(word_id)
            r_token_type_id.append(token_type_ids[batch_id][idx])
        after_dup_len = len(r_input_id)
        repetitied_input_ids.append(r_input_id)
        repetitied_token_type_ids.append(r_token_type_id)
        rep_seq_len = max(rep_seq_len, after_dup_len)

    # Padding the data to the same length
    for batch_id in range(batch_size):
        after_dup_len = len(repetitied_input_ids[batch_id])
        pad_len = rep_seq_len - after_dup_len
        repetitied_input_ids[batch_id] += [0] * pad_len
        repetitied_token_type_ids[batch_id] += [0] * pad_len

    return paddle.to_tensor(repetitied_input_ids), paddle.to_tensor(repetitied_token_type_ids)

