# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 Google Research and The HuggingFace Inc. team.
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

import numpy as np
import sentencepiece as spm

from paddlenlp.data.vocab import Vocab

from ..albert.tokenizer import AlbertEnglishTokenizer

__all__ = ["BigBirdTokenizer"]

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"bigbird-base-uncased": 4096}


class BigBirdTokenizer(AlbertEnglishTokenizer):
    """
    Constructs an BigBird tokenizer based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        sentencepiece_model_file (str):
            The vocabulary file (ends with '.spm') required to instantiate
            a `SentencePiece <https://github.com/google/sentencepiece>`__ tokenizer.
        do_lower_case (bool): Whether the text strips accents and convert to
            Whether or not to lowercase the input when tokenizing.
            Defaults to`True`.
        unk_token (str):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "[UNK]".
        sep_token (str):
            A special token separating two different sentences in the same input.
            Defaults to "[SEP]".
        pad_token (str):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "[PAD]".
        cls_token (str):
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to "[CLS]".
        mask_token (str):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to "[MASK]".

    Raises:
        ValueError: If file sentencepiece_model_file doesn't exist.

    """

    resource_files_names = {
        "sentencepiece_model_file": "sentencepiece_gpt2.model",
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "sentencepiece_model_file": {
            "bigbird-base-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/bigbird/sentencepiece_gpt2.model",
        },
    }
    pretrained_init_configuration = {
        "bigbird-base-uncased": {"do_lower_case": False},
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
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        extra_ids=100,
        additional_special_tokens=[],
        sp_model_kwargs=None,
        encoding="utf8",
        **kwargs
    ):

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.extra_ids = extra_ids
        self.sentencepiece_model_file = sentencepiece_model_file

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(sentencepiece_model_file)
        self.encoding = encoding
        vocab_dict = {}
        for id in range(self.sp_model.get_piece_size()):
            vocab_dict[self.sp_model.id_to_piece(id)] = id
        vocab_ = Vocab.from_dict(vocab_dict, unk_token=unk_token)
        self.start_word_tokens = np.array([vocab_._idx_to_token[i][0] == "▁" for i in range(0, len(vocab_))])

        self.unk_token = unk_token
        self.mask_id = vocab_dict[mask_token] if mask_token in vocab_dict else 0
        self.unk_id = vocab_dict[unk_token] if unk_token in vocab_dict else 0
        self.cls_id = vocab_dict[cls_token] if cls_token in vocab_dict else 0
        self.sep_id = vocab_dict[sep_token] if sep_token in vocab_dict else 0
        self.pad_id = vocab_dict[pad_token] if pad_token in vocab_dict else 0

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

        return super(BigBirdTokenizer, self).__call__(
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

        An BigBird sequence has the following format:

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

    def _encode(self, text, max_seq_len=None, max_pred_len=None, masked_lm_prob=0.15):
        """
        Returns a tuple containing the encoded sequence and mask information.

        Args:
            text (str,list[str] or list[int]):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            max_seq_len (int, optional):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If set to None, will not limit the total sequence.
                Defaults to None.
            max_pred_len (int, optional):
                If set to a number, will limit the mask sequence returned so that it has a maximum prediction length.
                If set to None, will not limit the mask sequence.
            masked_lm_prob (float, optional):
                The probability of the token to be masked. Defaults to `0.15`.
        Returns:
            tuple: Returns tuple (span_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights).
        """

        def get_input_ids(text):
            if isinstance(text, str):
                text = re.sub("[\n]+", "", text)
                tokens = self._tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        ids = get_input_ids(text)
        # Find the span for in the text
        max_seq_len = len(ids) if max_seq_len is None else max_seq_len
        max_pred_len = len(ids) if max_pred_len is None else max_pred_len

        end_pos = max_seq_len - 2 + np.random.randint(max(1, len(ids) - max_seq_len - 2))
        start_pos = max(0, end_pos - max_seq_len + 2)
        span_ids = ids[start_pos:end_pos]

        word_begin_flag = self.start_word_tokens[span_ids]
        word_begin_pos = np.flatnonzero(word_begin_flag).astype(np.int32)
        if word_begin_pos.size == 0:
            word_begin_pos = np.arange(len(span_ids), dtype=np.int32)
            word_begin_flag = np.logical_not(word_begin_flag)

        first_start_pos = word_begin_pos[0]
        span_ids = span_ids[first_start_pos:]
        num_tokens = len(span_ids)
        word_begin_pos = word_begin_pos - first_start_pos
        words = np.split(np.arange(len(span_ids), dtype="int32"), word_begin_pos)[1:]
        assert len(words) == len(word_begin_pos)
        num_to_predict = min(max_pred_len, max(1, int(round(len(word_begin_pos) * masked_lm_prob))))

        masked_lm_positions = np.concatenate(
            np.random.choice(np.array([[]] + words, dtype=np.object)[1:], num_to_predict, replace=False), 0
        )
        if len(masked_lm_positions) > max_pred_len:
            masked_lm_positions = masked_lm_positions[: max_pred_len + 1]
            truncate_masking_flag = np.flatnonzero(word_begin_flag[masked_lm_positions])
            if len(truncate_masking_flag) == 0:
                truncate_masking_index = max_pred_len
            else:
                truncate_masking_index = truncate_masking_flag[-1]
            masked_lm_positions = masked_lm_positions[:truncate_masking_index]
        span_ids = np.array(span_ids, dtype="int32")
        masked_lm_positions = np.sort(masked_lm_positions)
        masked_lm_ids = np.array(span_ids)[masked_lm_positions]

        random_prob = np.random.rand(len(masked_lm_positions))
        mask_pos = masked_lm_positions[random_prob < 0.8]
        random_pos = masked_lm_positions[random_prob > 0.9]
        span_ids[mask_pos] = self.mask_id
        span_ids[random_pos] = np.random.randint(self.unk_id + 1, self.vocab_size, len(random_pos), dtype=np.int32)
        span_ids = np.concatenate(
            [np.array([self.cls_id], dtype=np.int32), span_ids, np.array([self.sep_id], dtype=np.int32)]
        )
        padding_len = max_seq_len - num_tokens - 2
        span_ids = np.pad(span_ids, [0, padding_len], "constant")
        pred_padding_len = max_pred_len - len(masked_lm_positions)
        masked_lm_weights = np.pad(
            np.ones_like(masked_lm_positions, dtype=np.float32), [0, pred_padding_len], "constant"
        )
        masked_lm_positions = np.pad(masked_lm_positions + 1, [0, pred_padding_len], "constant")
        masked_lm_ids = np.pad(masked_lm_ids, [0, pred_padding_len], "constant")
        return span_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights
