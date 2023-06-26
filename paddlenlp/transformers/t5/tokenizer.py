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

import re
import warnings

import sentencepiece as spm

from ..albert.tokenizer import AlbertEnglishTokenizer

__all__ = [
    "T5Tokenizer",
]

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "t5-small": 512,
    "t5-base": 512,
    "t5-large": 512,
    "t5-3b": 512,
    "t5-11b": 512,
}


class T5Tokenizer(AlbertEnglishTokenizer):
    """
    Constructs a T5 tokenizer based on SentencePiece .
    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        sentencepiece_model_file (str):
            The vocabulary file (ends with '.spm') required to instantiate
            a `SentencePiece <https://github.com/google/sentencepiece>`__ tokenizer.
        do_lower_case (bool):
            Whether or not to lowercase the input when tokenizing. Defaults to `False`.
        remove_space (bool):
            Whether or note to remove space when tokenizing. Defaults to `True`.
        keep_accents (bool):
            Whether or note to keep accents when tokenizing. Defaults to `False`.
        eos_token (str):
            A special token representing the *eos (end-of-sentence)* token.
            Defaults to "</s>".
        unk_token (str):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "<unk>".
        pad_token (str):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "<pad>".

    """

    resource_files_names = {"sentencepiece_model_file": "spiece.model"}
    pretrained_resource_files_map = {
        "sentencepiece_model_file": {
            "t5-small": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-small/spiece.model",
            "t5-base": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-base/spiece.model",
            "t5-large": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-large/spiece.model",
            "t5-3b": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-3b/spiece.model",
            "t5-11b": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-11b/spiece.model",
            "t5-v1_1-base": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-v1_1-base/spiece.model",
            "t5-v1_1-large": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-v1_1-large/spiece.model",
        },
    }

    pretrained_init_configuration = {
        "t5-small": {"do_lower_case": False},
        "t5-base": {"do_lower_case": False},
        "t5-large": {"do_lower_case": False},
        "t5-3b": {"do_lower_case": False},
        "t5-11b": {"do_lower_case": False},
        "t5-v1_1-base": {"do_lower_case": False},
        "t5-v1_1-large": {"do_lower_case": False},
    }

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        sentencepiece_model_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=True,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=[],
        sp_model_kwargs=None,
        **kwargs
    ):

        # Add extra_ids to the special token list
        if extra_ids > 0 and len(additional_special_tokens) == 0:
            self._additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and len(additional_special_tokens) != 0:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in str(x)), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.extra_ids = extra_ids
        self.sentencepiece_model_file = sentencepiece_model_file

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(sentencepiece_model_file)

    def __call__(
        self,
        text,
        text_pair=None,
        max_length=None,
        stride=0,
        is_split_into_words=False,
        padding=None,
        truncation="longest_first",
        return_position_ids=False,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        **kwargs
    ):
        if "pad_to_max_seq_len" in kwargs and padding is None:
            pad_to_max_seq_len = kwargs.pop("pad_to_max_seq_len")
            padding = "max_length" if pad_to_max_seq_len else False
        elif padding is None:
            padding = False

        if "max_seq_len" in kwargs and max_length is None:
            max_length = kwargs["max_seq_len"]

        if "truncation_strategy" in kwargs and kwargs["truncation_strategy"] != "longest_first":
            truncation = kwargs["truncation_strategy"]

        return super(T5Tokenizer, self).__call__(
            text=text,
            text_pair=text_pair,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            padding=padding,
            truncation=truncation,
            return_position_ids=return_position_ids,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_length=return_length,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.sp_model) + self.extra_ids

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
        """
        Build model inputs from a sequence or a pair of sequence.

        An Reformer sequence has the following format:

        - single sequence:      ``X </s>``
        - pair of sequences:        ``A </s> B </s>``

        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs. Defaults to None.

        Returns:
            List[int]: List of input_id with the appropriate special tokens.

        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def build_offset_mapping_with_special_tokens(self, offset_mapping_0, offset_mapping_1=None):
        """
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens.

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            offset_mapping_0 (List[tuple]):
                List of char offsets to which the special tokens will be added.
            offset_mapping_1 (List[tuple], optional):
                Optional second list of char offsets for offset mapping pairs.

        Returns:
            List[tuple]: List of char offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return offset_mapping_0 + [(0, 0)]

        return offset_mapping_0 + [(0, 0)] + offset_mapping_1 + [(0, 0)]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences.

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (List[int]):
                List of IDs.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.

        Returns:
            List[int]: List of token_type_id according to the given sequence(s).

        """
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.

        Args:
            token_ids_0 (List[int]): List of ids of the first sequence.
            token_ids_1 (List[int], optional): List of ids of the second sequence.
            already_has_special_tokens (bool, optional): Whether or not the token list is already
                formatted with special tokens for the model. Defaults to None.

        Returns:
            List[int]: The list of integers in the range [0, 1]:
                1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

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
                out_string += self.sp_model.decode_pieces(current_sub_tokens) + token + " "
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode_pieces(current_sub_tokens)
        return out_string.strip()

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            token = f"<extra_id_{self.vocab_size - 1 - index}>"
        return token

    def batch_decode(self, sequences, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (Union[List[int], List[List[int]], Tensor]):
                List of tokenized input ids.
            skip_special_tokens (bool, optional):
                Whether or not to remove special tokens in the decoding. Defaults to `False`.
            clean_up_tokenization_spaces (bool, optional):
                Whether or not to clean up the tokenization spaces. Defaults to `True`.

        Returns:
            List[str]: The list of decoded sentences.
        """
        return [
            self.decode(
                seq, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
            for seq in sequences
        ]

    @staticmethod
    def clean_up_tokenization(out_string):
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (str): The text to clean up.

        Returns:
            str: The cleaned-up string.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.sentencepiece_model_file)
