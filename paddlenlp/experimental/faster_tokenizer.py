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
import paddle.fluid.core as core
import paddle.nn as nn
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import in_dygraph_mode

__all__ = ["to_tensor", "to_vocab_buffer", "FasterTokenizer"]


def to_tensor(string_values, name="text"):
    """
    Create the tensor that the value holds the list of string.
    NOTICE: The value will be holded in the cpu place. 
 
    Args:
        string_values(list[string]): The value will be setted to the tensor.
        name(string): The name of the tensor.
    """
    tensor = paddle.Tensor(core.VarDesc.VarType.STRING, [], name,
                           core.VarDesc.VarType.STRINGS, False)
    tensor.value().set_string_list(string_values)
    return tensor


def to_vocab_buffer(vocab_dict, name):
    """
    Create the tensor that the value holds the map, the type of key is the string.
    NOTICE: The value will be holded in the cpu place. 
 
    Args:
        vocab_dict(dict): The value will be setted to the tensor. 
            The key is token and the value is the token index.
        name(string): The name of the tensor.
    """
    tensor = paddle.Tensor(core.VarDesc.VarType.RAW, [], name,
                           core.VarDesc.VarType.VOCAB, True)
    tensor.value().set_vocab(vocab_dict)
    return tensor


class FasterTokenizer(nn.Layer):
    def __init__(self, vocab, do_lower_case=False, is_split_into_words=False):
        super(FasterTokenizer, self).__init__()
        vocab_buffer = to_vocab_buffer(vocab, "vocab")
        self.register_buffer("vocab", vocab_buffer, persistable=True)

        self.do_lower_case = do_lower_case
        self.is_split_into_words = is_split_into_words

    def forward(self,
                text,
                text_pair=None,
                max_seq_len=-1,
                pad_to_max_seq_len=False):
        if in_dygraph_mode():
            input_ids, seg_ids = core.ops.faster_tokenizer(
                self.vocab, text, text_pair, "do_lower_case",
                self.do_lower_case, "max_seq_len", max_seq_len,
                "pad_to_max_seq_len", pad_to_max_seq_len, "is_split_into_words",
                self.is_split_into_words)
            return input_ids, seg_ids

        attrs = {
            "do_lower_case": self.do_lower_case,
            "max_seq_len": max_seq_len,
            "pad_to_max_seq_len": pad_to_max_seq_len,
            "is_split_into_words": self.is_split_into_words,
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
                type='faster_tokenizer',
                inputs={
                    'Vocab': self.vocab,
                    'Text': text,
                    'TextPair': text_pair
                },
                outputs={'InputIds': input_ids,
                         'SegmentIds': seg_ids},
                attrs=attrs)
        return input_ids, seg_ids
