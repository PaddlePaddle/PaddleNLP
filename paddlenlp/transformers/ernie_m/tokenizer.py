# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
from typing import List, Optional

import sentencepiece as spm

from .. import PretrainedTokenizer

__all__ = ["ErnieMTokenizer"]

SPIECE_UNDERLINE = "▁"


class ErnieMTokenizer(PretrainedTokenizer):
    r"""
    Constructs a ErnieM tokenizer. It uses the `sentencepiece` tools to cut the words to sub-words.

    Args:
        vocab_file (str):
            The file path of the vocabulary.
        sentencepiece_model_file (str):
            The file path of sentencepiece model.
        do_lower_case (str, optional):
            Whether or not to lowercase the input when tokenizing.
            Defaults to`True`.
        unk_token (str, optional):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "[UNK]".
        sep_token (str, optional):
            A special token separating two different sentences in the same input.
            Defaults to "[SEP]".
        pad_token (str, optional):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "[PAD]".
        cls_token (str, optional):
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to "[CLS]".
        mask_token (str, optional):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to "[MASK]".
    """
    resource_files_names = {
        "sentencepiece_model_file": "sentencepiece.bpe.model",
        "vocab_file": "vocab.txt",
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-m-base": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.vocab.txt",
            "ernie-m-large": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.vocab.txt",
            "uie-m-base": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.vocab.txt",
            "uie-m-large": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.vocab.txt",
        },
        "sentencepiece_model_file": {
            "ernie-m-base": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.sentencepiece.bpe.model",
            "ernie-m-large": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.sentencepiece.bpe.model",
            "uie-m-base": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.sentencepiece.bpe.model",
            "uie-m-large": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_m/ernie_m.sentencepiece.bpe.model",
        },
    }
    pretrained_init_configuration = {
        "ernie-m-base": {"do_lower_case": False},
        "ernie-m-large": {"do_lower_case": False},
        "uie-m-base": {"do_lower_case": False},
        "uie-m-large": {"do_lower_case": False},
    }
    max_model_input_sizes = {"ernie-m-base": 514, "ernie-m-large": 514, "uie-m-base": 514, "uie-m-large": 514}
    # Ernie-M model doesn't have token_type embedding.
    model_input_names: List[str] = ["input_ids"]

    def __init__(
        self,
        vocab_file,
        sentencepiece_model_file,
        do_lower_case=False,
        encoding="utf8",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        self.sp_model = spm.SentencePieceProcessor()

        self.do_lower_case = do_lower_case
        self.encoding = encoding
        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocabulary file at path '{}'.".format(vocab_file))
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
        self.vocab_file = vocab_file
        self.sentencepiece_model_file = sentencepiece_model_file
        if os.path.isfile(sentencepiece_model_file):
            self.sp_model.Load(sentencepiece_model_file)

        self.SP_CHAR_MAPPING = {}

        for ch in range(65281, 65375):
            if ch in [ord("～")]:
                self.SP_CHAR_MAPPING[chr(ch)] = chr(ch)
                continue
            self.SP_CHAR_MAPPING[chr(ch)] = chr(ch - 65248)

    def get_offset_mapping(self, text):
        if text is None:
            return None

        split_tokens = self.tokenize(text)
        normalized_text, char_mapping = "", []

        for i, ch in enumerate(text):

            if ch in self.SP_CHAR_MAPPING:
                ch = self.SP_CHAR_MAPPING.get(ch)
            else:
                ch = unicodedata.normalize("NFKC", ch)
            if self.is_whitespace(ch):
                continue
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0

        if self.do_lower_case:
            text = text.lower()

        for token in split_tokens:
            if token[:1] == "▁":
                token = token[1:]
            start = text[offset:].index(token) + offset
            end = start + len(token)

            token_mapping.append((char_mapping[start], char_mapping[end - 1] + 1))
            offset = end
        return token_mapping

    @property
    def vocab_size(self):
        r"""
        Return the size of vocabulary.

        Returns:
            int: The size of vocabulary.
        """
        return self.sp_model.vocab_size()

    def get_vocab(self):
        return dict(self.vocab.token_to_idx, **self.added_tokens_encoder)

    def clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        return "".join((self.SP_CHAR_MAPPING.get(c, c) for c in text))

    def _tokenize(self, text, sample=False):
        """Tokenize a string."""
        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if piece == SPIECE_UNDERLINE:
                continue
            lst_i = 0
            for i, c in enumerate(piece):
                if c == SPIECE_UNDERLINE:
                    continue
                if self.is_ch_char(c) or self.is_punct(c):
                    if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                        new_pieces.append(piece[lst_i:i])
                    new_pieces.append(c)
                    lst_i = i + 1
                elif c.isdigit() and i > 0 and not piece[i - 1].isdigit():
                    if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                        new_pieces.append(piece[lst_i:i])
                    lst_i = i
                elif not c.isdigit() and i > 0 and piece[i - 1].isdigit():
                    if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                        new_pieces.append(piece[lst_i:i])
                    lst_i = i
            if len(piece) > lst_i:
                new_pieces.append(piece[lst_i:])
        return new_pieces

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def convert_ids_to_string(self, ids):
        """
        Converts a sequence of tokens (strings for sub-words) in a single string.
        """
        tokens = self.convert_ids_to_tokens(ids)
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        r"""
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        An ERNIE-M sequence has the following format:
        - single sequence:       ``[CLS] X [SEP]``
        - pair of sequences:        ``[CLS] A [SEP] [SEP] B [SEP]``
        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.
                Defaults to `None`.
        Returns:
            List[int]: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + _sep + token_ids_1 + _sep

    def build_offset_mapping_with_special_tokens(self, offset_mapping_0, offset_mapping_1=None):
        r"""
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens.

        An ERNIE-M offset_mapping has the following format:
        - single sequence:      ``(0,0) X (0,0)``
        - pair of sequences:        ``(0,0) A (0,0) (0,0) B (0,0)``

        Args:
            offset_mapping_ids_0 (List[tuple]):
                List of char offsets to which the special tokens will be added.
            offset_mapping_ids_1 (List[tuple], optional):
                Optional second list of wordpiece offsets for offset mapping pairs.
                Defaults to `None`.
        Returns:
            List[tuple]: List of wordpiece offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0), (0, 0)] + offset_mapping_1 + [(0, 0)]

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        r"""
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.
        Args:
            token_ids_0 (List[int]):
                List of ids of the first sequence.
            token_ids_1 (List[int], optinal):
                Optional second list of IDs for sequence pairs.
                Defaults to `None`.
            already_has_special_tokens (str, optional):
                Whether or not the token list is already formatted with special tokens for the model.
                Defaults to `False`.
        Returns:
            List[int]:
                The list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids)

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The token type ids.
        """
        # called when `add_special_tokens` is True, so align with `build_inputs_with_special_tokens` method
        if token_ids_1 is None:
            # [CLS] X [SEP]
            return (len(token_ids_0) + 2) * [0]

        # [CLS] A [SEP] [SEP] B [SEP]
        return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 3)

    def is_ch_char(self, char):
        """
        is_ch_char
        """
        if "\u4e00" <= char <= "\u9fff":
            return True
        return False

    def is_alpha(self, char):
        """
        is_alpha
        """
        if "a" <= char <= "z":
            return True
        if "A" <= char <= "Z":
            return True
        return False

    def is_punct(self, char):
        """
        is_punct
        """
        if char in ",;:.?!~，；：。？！《》【】":
            return True
        return False

    def is_whitespace(self, char):
        """
        is whitespace
        """
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        if len(char) == 1:
            cat = unicodedata.category(char)
            if cat == "Zs":
                return True
        return False
