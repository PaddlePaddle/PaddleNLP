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

import os
import unicodedata
from shutil import copyfile
from typing import List, Optional

from paddle.utils import try_import

from .. import PretrainedTokenizer

__all__ = ['FNetTokenizer']

SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility


class FNetTokenizer(PretrainedTokenizer):
    """
    Construct an FNet tokenizer. Adapted from :class:`~transformers.AlbertTokenizer`. Based on `SentencePiece
    <https://github.com/google/sentencepiece>`__. This tokenizer inherits from
    :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods. Users should refer to this
    superclass for more information regarding those methods.

    Args:
        vocab_file (str):
            The vocabulary file (ends with '.spm') required to instantiate
            a `SentencePiece <https://github.com/google/sentencepiece>`__ tokenizer.
        do_lower_case (bool, optional):
            Whether or not to lowercase the input when tokenizing. Defaults to `False` and
            **does not** lowercase the input.
        remove_space (bool, optional):
            Whether or not to strip the text when tokenizing. Defaults to `True` and
            removes excess spaces before and after the string.
        keep_accents (bool, optional):
            Whether or not to keep accents when tokenizing. Defaults to `True` and **does not** keep accents.
        unk_token (str, optional):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to `"<unk>"`.
        sep_token (str, optional):
            A special token separating two different sentences in the same input.
            Defaults to `"[SEP]"`.
        pad_token (str, optional):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to `"<pad>"`.
        cls_token (str, optional):
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to `"[CLS]"`.
        mask_token (str, optional):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to `"[MASK]"`.

    Attributes:
        sp_model (SentencePieceProcessor):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).

    """
    resource_files_names = {"vocab_file": "spiece.model"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "fnet-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/fnet/fnet-base/spiece.model",
            "fnet-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/fnet/fnet-large/spiece.model",
        },
    }

    pretrained_init_configuration = {
        "fnet-base": {
            "do_lower_case": False
        },
        "fnet-large": {
            "do_lower_case": False
        },
    }

    pretrained_positional_embedding_sizes = {
        "fnet-base": None,
        "fnet-large": None
    }

    def __init__(self,
                 vocab_file,
                 do_lower_case=False,
                 remove_space=True,
                 keep_accents=True,
                 unk_token="<unk>",
                 sep_token="[SEP]",
                 pad_token="<pad>",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 **kwargs):

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
        # """
        # Converts a string to a list of tokens.
        #
        # Args:
        #     text (str):
        #         The text to be tokenized.
        # Returns:
        #     List(str): A list of string representing converted tokens.
        # """
        return self._tokenize(text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) to an id using the vocab. """
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) to a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a token (or a sequence of tokens) to a single integer id (or a sequence of ids),
        using the vocabulary.

        Args:
            tokens (str or List[str]):
                One or several token(s) to convert to token id(s).

        Returns:
            int or List[int] or tuple(int): The token id or list of token ids or tuple of token ids.
        """
        if not isinstance(tokens, (list, tuple)):
            return self._convert_token_to_id(tokens)
        else:
            return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """
        Converts a single index or a sequence of indices to a token or
        a sequence of tokens, using the vocabulary and added tokens.

        Args:
            ids (int or List[int]):
                The token id (or token ids) to be converted to token(s).
            skip_special_tokens (bool, optional):
                Whether or not to remove special tokens in the decoding.
                Defaults to `False` and we do not remove special tokens.

        Returns:
            str or List[str]: The decoded token(s).
        """
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
        return self.sp_model.decode(tokens)

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An FNet sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True)

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An FNet sequence
        pair mask has the following format: ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 | first sequence | second sequence |

        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_resources(self, save_directory):
        """
        Saves `SentencePiece <https://github.com/google/sentencepiece>`__ file
        (ends with '.spm') under `save_directory`.

        Args:
            save_directory (str):
                Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            save_path = os.path.join(save_directory, file_name)
            if os.path.abspath(self.vocab_file) != os.path.abspath(save_path):
                copyfile(self.vocab_file, save_path)