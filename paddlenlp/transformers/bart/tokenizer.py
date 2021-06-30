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
from .. import GPTTokenizer

__all__ = ['BartTokenizer']


class BartTokenizer(GPTTokenizer):
    r"""
    Construct a BART tokenizer.
    """
    # merges and vocab same as GPT2
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt"
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "bart-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bart/bart-base-vocab.json",
            "bart-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bart/bart-large-vocab.json",
        },
        "merges_file": {
            "bart-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bart/bart-base-merges.txt",
            "bart-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bart/bart-large-merges.txt",
        }
    }
    pretrained_init_configuration = {"bart-base": {}, "bart-large": {}}

    def __init__(
            self,
            vocab_file,
            merges_file,
            errors='replace',
            max_len=None,
            special_tokens=None,
            bos_token="<s>",
            eos_token="</s>",
            cls_token="<s>",
            sep_token="</s>",
            pad_token="<pad>",
            eol_token='\u010a',  # The token of newline.
    ):
        super(BartTokenizer, self).__init__(vocab_file, merges_file, errors,
                                            max_len, special_tokens, pad_token,
                                            eos_token, eol_token)

    def __call__(self,
                 text,
                 text_pair=None,
                 max_seq_len=None,
                 stride=0,
                 is_split_into_words=False,
                 pad_to_max_seq_len=False,
                 truncation_strategy="longest_first",
                 return_position_ids=False,
                 return_token_type_ids=False,
                 return_attention_mask=True,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        return super(BartTokenizer, self).__call__(
            text, text_pair, max_seq_len, stride, is_split_into_words,
            pad_to_max_seq_len, truncation_strategy, return_position_ids,
            return_token_type_ids, return_attention_mask, return_length,
            return_overflowing_tokens, return_special_tokens_mask)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        if token_ids_1 is None:
            return _cls + token_ids_0 + _sep
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep
