# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

from paddle.utils import try_import
from ..albert.tokenizer import AlbertEnglishTokenizer
import warnings


class T5Tokenizer(AlbertEnglishTokenizer):
    resource_files_names = {"sentencepiece_model_file": "spiece.model", }
    pretrained_resource_files_map = {
        "sentencepiece_model_file": {
            "t5-small":
            "https://huggingface.co/t5-small/resolve/main/spiece.model",
            "t5-base":
            "https://huggingface.co/t5-base/resolve/main/spiece.model",
            "t5-large":
            "https://huggingface.co/t5-large/resolve/main/spiece.model"
        },
    }

    pretrained_init_configuration = {
        "t5-small": {
            "do_lower_case": False
        },
        "t5-base": {
            "do_lower_case": False
        },
        "t5-large": {
            "do_lower_case": False
        },
    }

    def __init__(self,
                 sentencepiece_model_file,
                 do_lower_case=False,
                 remove_space=True,
                 keep_accents=False,
                 eos_token="</s>",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 **kwargs):

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.sentencepiece_model_file = sentencepiece_model_file

        spm = try_import("sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sentencepiece_model_file)

    def _add_eos_if_not_present(self, token_ids):
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1):
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def get_special_tokens_mask(
            self,
            token_ids_0,
            token_ids_1=None,
            already_has_special_tokens=False, ):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True, )

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += (self.sp_model.decode_pieces(current_sub_tokens) +
                               token + " ")
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode_pieces(current_sub_tokens)
        return out_string.strip()

    def decode(self,
               seq,
               skip_special_tokens=False,
               clean_up_tokenization_spaces=True):
        if hasattr(seq, "tolist"):
            seq = seq.tolist()
        text = self.convert_tokens_to_string(
            self.convert_ids_to_tokens(
                seq, skip_special_tokens=skip_special_tokens))
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)
        return text

    def batch_decode(
            self,
            sequences,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True, ):
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces)
            for seq in sequences
        ]

    def clean_up_tokenization(self, out_string):
        out_string = (out_string.replace(" .", ".").replace(" ?", "?")
                      .replace(" !", "!").replace(" ,", ",").replace(" ' ", "'")
                      .replace(" n't", "n't").replace(" 'm", "'m")
                      .replace(" 's", "'s").replace(" 've", "'ve")
                      .replace(" 're", "'re"))
        return out_string
