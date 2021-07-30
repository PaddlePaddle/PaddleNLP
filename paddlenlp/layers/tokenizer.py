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
import io

import paddle
import paddle.fluid.core as core
import paddle.nn as nn
import paddlenlp
from paddlenlp.transformers import BertTokenizer

__all__ = ["Tokenizer"]


def load_vocabulary(filepath):
    """
    load vocab
    """
    token_to_idx = {}
    with io.open(filepath, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.rstrip('\n')
            token_to_idx[token] = int(index)
    return token_to_idx


class Tokenizer(nn.Layer):
    def __init__(self, vocab_path):
        super(Tokenizer, self).__init__()
        vocab = load_vocabulary(vocab_path)
        vocab_tensor = paddlenlp.ops.to_map_tensor(vocab, "demo_vocab")
        self.register_buffer("vocab", vocab_tensor, persistable=True)

    def forward(self,
                text,
                text_pair=None,
                max_seq_len=None,
                stride=0,
                is_split_into_words=False,
                pad_to_max_seq_len=False,
                truncation_strategy="longest_first",
                return_position_ids=False,
                return_token_type_ids=True,
                return_attention_mask=False,
                return_length=False,
                return_overflowing_tokens=False,
                return_special_tokens_mask=False):
        tokens_tensor = paddlenlp.ops.to_strings_tensor(text, "demo_tokens")
        input_ids, seg_ids = core.ops.tokenizer(
            tokens_tensor,
            self.vocab,
            text_pair,
            max_seq_len=max_seq_len,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_max_seq_len=pad_to_max_seq_len,
            truncation_strategy=truncation_strategy,
            return_position_ids=return_position_ids,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_length=return_length,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask)
        return input_ids, seg_ids
