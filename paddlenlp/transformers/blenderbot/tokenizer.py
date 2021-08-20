# encoding=utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .. import GPTTokenizer
from paddle.utils import try_import

__all__ = ['BlenderbotTokenizer']


class BlenderbotTokenizer(GPTTokenizer):
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt"
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "blenderbot-400M-distill":
                "https://paddlenlp.bj.bcebos.com/models/transformers/blenderbot/blenderbot-400M-distill-vocab.json",
            "blenderbot-3B":
                "https://paddlenlp.bj.bcebos.com/models/transformers/blenderbot/blenderbot-3B-vocab.json",
            "blenderbot-1B-distill":
                "https://paddlenlp.bj.bcebos.com/models/transformers/blenderbot/blenderbot-1B-distill-vocab.json"},
        "merges_file": {
            "blenderbot-400M-distill":
                "https://paddlenlp.bj.bcebos.com/models/transformers/blenderbot/blenderbot-400M-distill-merges.txt",
            "blenderbot-3B":
                "https://paddlenlp.bj.bcebos.com/models/transformers/blenderbot/blenderbot-3B-merges.txt",
            "blenderbot-1B-distill":
                "https://paddlenlp.bj.bcebos.com/models/transformers/blenderbot/blenderbot-1B-distill-merges.txt"
        }
    }
    pretrained_init_configuration = {"blenderbot-3B": {"add_prefix": True},
                                     "blenderbot-400M-distill": {"add_prefix": True},
                                     "blenderbot-1B-distill": {"add_prefix": True}}

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
            add_prefix=True,  # Add " " before text for tokenize
            eol_token='\u010a',  # The token of newline.
    ):
        super(BlenderbotTokenizer, self).__init__(vocab_file, merges_file, errors,
                                                  max_len, special_tokens, pad_token,
                                                  eos_token, eol_token)
        self.add_prefix = add_prefix

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
        return super(BlenderbotTokenizer, self).__call__(
            text, text_pair, max_seq_len, stride, is_split_into_words,
            pad_to_max_seq_len, truncation_strategy, return_position_ids,
            return_token_type_ids, return_attention_mask, return_length,
            return_overflowing_tokens, return_special_tokens_mask)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Format of Blenderbot sequence: ``X </s>``
        :param token_ids_0: List[int]
        :param token_ids_1: List[int], optional
        :return: List[int]
        """
        return token_ids_0 + [self.eos_token_id]

    def _tokenize(self, text):
        """ Tokenize a string. """
        if self.add_prefix:
            text = " " + text
        bpe_tokens = []
        re = try_import("regex")
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(
                bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

