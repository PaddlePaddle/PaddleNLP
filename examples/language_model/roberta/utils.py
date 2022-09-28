# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
from paddlenlp.data import Stack, Tuple, Pad, Dict


class DataCollatorMLM():

    def __init__(self, tokenizer, batch_pad=None):
        self.batch_pad = batch_pad
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.token_len = tokenizer.vocab_size
        if batch_pad is None:
            self.batch_pad = lambda samples, fn=Dict({
                'input_ids':
                Pad(axis=0, pad_val=self.pad_token_id, dtype='int64'),  # input
                # 'token_type_ids': Pad(axis=0, pad_val=0, dtype='int64'),  # segment
                'special_tokens_mask':
                Pad(axis=0, pad_val=True, dtype='int64')  # segment
            }): fn(samples)
        else:
            self.batch_pad = batch_pad

    def __call__(self, examples):
        examples = self.batch_pad(examples)
        examples = [paddle.to_tensor(e) for e in examples]
        examples[0], labels = self._mask_tokens(
            examples[0], paddle.cast(examples[1], dtype=bool),
            self.mask_token_id, self.token_len)
        examples.append(labels)
        return examples

    def _mask_tokens(self,
                     inputs,
                     special_tokens_mask,
                     mask_token_id,
                     token_len,
                     mlm_prob=0.15,
                     ignore_label=-100):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        probability_matrix = paddle.full(labels.shape, mlm_prob)
        probability_matrix[special_tokens_mask] = 0

        masked_indices = paddle.cast(paddle.bernoulli(probability_matrix),
                                     dtype=bool)
        labels[
            ~masked_indices] = ignore_label  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = paddle.cast(paddle.bernoulli(
            paddle.full(labels.shape, 0.8)),
                                       dtype=bool) & masked_indices
        inputs[indices_replaced] = mask_token_id

        # 10% of the time, we replace masked input tokens with random word

        indices_random = paddle.cast(
            paddle.bernoulli(paddle.full(labels.shape, 0.5)),
            dtype=bool) & masked_indices & ~indices_replaced
        random_words = paddle.randint(low=0, high=token_len, shape=labels.shape)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
