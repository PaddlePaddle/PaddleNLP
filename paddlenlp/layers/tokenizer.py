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

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import in_dygraph_mode
from paddlenlp.ops import to_vocab_tensor

__all__ = ["FastTokenizer"]


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


class FasterTokenizer(nn.Layer):
    def __init__(self, vocab_path):
        super(FasterTokenizer, self).__init__()
        vocab_dict = load_vocabulary(vocab_path)
        vocab_tensor = to_vocab_tensor(vocab_dict, "vocab")
        self.register_buffer("vocab", vocab_tensor, persistable=True)

    def forward(self,
                text,
                text_pair=None,
                do_lower_case=True,
                max_seq_len=-1,
                is_split_into_words=False,
                pad_to_max_seq_len=False):
        if in_dygraph_mode():
            input_ids, seg_ids = core.ops.faster_tokenizer(
                self.vocab, text, text_pair, "do_lower_case", do_lower_case,
                "max_seq_len", max_seq_len, "pad_to_max_seq_len",
                pad_to_max_seq_len, "is_split_into_words", is_split_into_words)
            return input_ids, seg_ids

        attrs = {
            "do_lower_case": do_lower_case,
            "max_seq_len": max_seq_len,
            "pad_to_max_seq_len": pad_to_max_seq_len,
            "is_split_into_words": is_split_into_words,
        }
        helper = LayerHelper("faster_tokenizer")
        input_ids = helper.create_variable_for_type_inference(dtype="int64")
        seg_ids = helper.create_variable_for_type_inference(dtype="int64")
        if text_pair is None:
            helper.append_op(
                type='faster_tokenizer',
                inputs={'Vocab': self.vocab,
                        'Text': text},
                outputs={'InputIds': input_ids,
                         'SegmentIds': seg_ids},
                attrs=attrs)
        else:
            helper.append_op(
                type='tokenizer',
                inputs={
                    'Vocab': self.vocab,
                    'Text': text,
                    'TextPair': text_pair
                },
                outputs={'InputIds': input_ids,
                         'SegmentIds': seg_ids},
                attrs=attrs)
        return input_ids, seg_ids
