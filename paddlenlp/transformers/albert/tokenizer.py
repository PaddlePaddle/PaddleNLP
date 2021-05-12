# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
"""Tokenization class for ALBERT model."""

import os
import unicodedata
from shutil import copyfile

from paddle.utils import try_import
from .. import PretrainedTokenizer
from .. import BertTokenizer

__all__ = [
    'AlbertTokenizer',
    'AlbertChineseTokenizer',
]

SPIECE_UNDERLINE = "▁"


class AlbertTokenizer(PretrainedTokenizer):
    resource_files_names = {"vocab_file": "spiece.model"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "albert-base-v1":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-base-v1.spiece.model",
            "albert-large-v1":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-large-v1.spiece.model",
            "albert-xlarge-v1":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-xlarge-v1.spiece.model",
            "albert-xxlarge-v1":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-xxlarge-v1.spiece.model",
            "albert-base-v2":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-base-v2.spiece.model",
            "albert-large-v2":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-large-v2.spiece.model",
            "albert-xlarge-v2":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-xlarge-v2.spiece.model",
            "albert-xxlarge-v2":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-xxlarge-v2.spiece.model",
            "albert-chinese-tiny":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-chinese-tiny.vocab.txt",
            "albert-chinese-small":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-chinese-small.vocab.txt",
            "albert-chinese-base":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-chinese-base.vocab.txt",
            "albert-chinese-large":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-chinese-large.vocab.txt",
            "albert-chinese-xlarge":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-chinese-xlarge.vocab.txt",
            "albert-chinese-xxlarge":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-chinese-xxlarge.vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "albert-base-v1": {
            "do_lower_case": False
        },
        "albert-large-v1": {
            "do_lower_case": False
        },
        "albert-xlarge-v1": {
            "do_lower_case": False
        },
        "albert-xxlarge-v1": {
            "do_lower_case": False
        },
        "albert-base-v2": {
            "do_lower_case": False
        },
        "albert-large-v2": {
            "do_lower_case": False
        },
        "albert-xlarge-v2": {
            "do_lower_case": False
        },
        "albert-xxlarge-v2": {
            "do_lower_case": False
        },
        "albert-chinese-tiny": {
            "do_lower_case": False
        },
        "albert-chinese-small": {
            "do_lower_case": False
        },
        "albert-chinese-base": {
            "do_lower_case": False
        },
        "albert-chinese-large": {
            "do_lower_case": False
        },
        "albert-chinese-xlarge": {
            "do_lower_case": False
        },
        "albert-chinese-xxlarge": {
            "do_lower_case": False
        },
    }

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        remove_space=True,
        keep_accents=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        spm = try_import("sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {
            self.convert_ids_to_tokens(i): i
            for i in range(self.vocab_size)
        }
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        spm = try_import("sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join(
                [c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text, sample=False):
        """Tokenize a string."""
        text = self.preprocess_text(text)

        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(
                    SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][
                        0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def tokenize(self, text):
        return self._tokenize(text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) to an id using the vocab. """
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) to a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_ids(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._convert_token_to_id(tokens)
        else:
            return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if not isinstance(ids, (list, tuple)):
            return self._convert_id_to_token(ids)
        tokens = [self._convert_id_to_token(_id) for _id in ids]
        if skip_special_tokens:
            return [
                token for token in tokens
                if token not in self.all_special_tokens
            ]
        return tokens

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)] + offset_mapping_1 + [(0, 0)]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_resources(self, save_directory):
        for name, file_name in self.resource_files_names.items():
            save_path = os.path.join(save_directory, file_name)
            if os.path.abspath(self.vocab_file) != os.path.abspath(save_path):
                copyfile(self.vocab_file, save_path)


class AlbertChineseTokenizer(BertTokenizer):
    resource_files_names = {"vocab_file": "vocab.txt"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "albert-chinese-tiny":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-chinese-tiny.vocab.txt",
            "albert-chinese-small":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-chinese-small.vocab.txt",
            "albert-chinese-base":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-chinese-base.vocab.txt",
            "albert-chinese-large":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-chinese-large.vocab.txt",
            "albert-chinese-xlarge":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-chinese-xlarge.vocab.txt",
            "albert-chinese-xxlarge":
                "https://paddlenlp.bj.bcebos.com/models/transformers/albert/albert-chinese-xxlarge.vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "albert-chinese-tiny": {
            "do_lower_case": False
        },
        "albert-chinese-small": {
            "do_lower_case": False
        },
        "albert-chinese-base": {
            "do_lower_case": False
        },
        "albert-chinese-large": {
            "do_lower_case": False
        },
        "albert-chinese-xlarge": {
            "do_lower_case": False
        },
        "albert-chinese-xxlarge": {
            "do_lower_case": False
        },
    }
