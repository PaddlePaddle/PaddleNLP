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

import io
import os
import six
import re
import numpy as np
from paddle.utils import try_import
from paddlenlp.data import Vocab
from .. import PretrainedTokenizer

__all__ = ['BigBirdTokenizer']


class BigBirdTokenizer(PretrainedTokenizer):
    """
    Constructs a BigBird tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.

    Args:
        sentencepiece_model_file (`str`): File path of the vocabulary
        do_lower_case (`bool`): Whether the text strips accents and convert to
            lower case. If you use the BigBird pretrained model, lower is set to
            False when using the cased model, otherwise it is set to True.
            Defaults to True.
        unk_token (`str`): The special token for unkown words. Defaults to `[UNK]`.
        sep_token (`str`): The special token for separator token . Defaults to `[SEP]`.
        pad_token (`str`): The special token for padding. Defaults to `[PAD]`.
        cls_token (`str`): The special token for cls. Defaults to `[CLS]`.
        mask_token (`str`): The special token for mask. Defaults to `[MASK]`.
    
    Raises:
        ValueError: If file sentencepiece_model_file doesn't exist. 

    """
    resource_files_names = {
        "sentencepiece_model_file": "sentencepiece_gpt2.model",
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "sentencepiece_model_file": {
            "bigbird-base-uncased":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bigbird/sentencepiece_gpt2.model",
        },
    }
    pretrained_init_configuration = {
        "bigbird-base-uncased": {
            "do_lower_case": True
        },
    }

    def __init__(self,
                 sentencepiece_model_file,
                 do_lower_case=True,
                 encoding="utf8",
                 unk_token="<unk>",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]"):

        if not os.path.isfile(sentencepiece_model_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = BigBirdTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(sentencepiece_model_file))
        self.encoding = encoding
        mod = try_import('sentencepiece')
        self.sp_model = mod.SentencePieceProcessor()
        if os.path.isfile(sentencepiece_model_file):
            self.sp_model.Load(sentencepiece_model_file)
        vocab_dict = {}
        for id in range(self.sp_model.get_piece_size()):
            vocab_dict[self.sp_model.id_to_piece(id)] = id
        self.vocab = Vocab.from_dict(vocab_dict, unk_token=unk_token)
        self.start_word_tokens = np.array([
            self.vocab._idx_to_token[i][0] == "▁"
            for i in range(0, len(self.vocab))
        ])
        self.unk_token = unk_token
        self.mask_id = vocab_dict[mask_token]
        self.unk_id = vocab_dict[unk_token]
        self.cls_id = vocab_dict[cls_token]
        self.sep_id = vocab_dict[sep_token]
        self.pad_id = vocab_dict[pad_token] if pad_token in vocab_dict else 0

    @property
    def vocab_size(self):
        """
        Returns the size of vocabulary.

        Returns:
            `Int`: The size of vocabulary.
        """
        return len(self.vocab)

    def _tokenize(self, text):
        """
        End-to-end tokenization for BigBird models.

        Args:
            text (str): The text to be tokenized.
        
        Returns:
            `List`: A list of string representing converted tokens.
        """
        if len(text) == 0:
            return []
        if not isinstance(text, six.string_types):
            text = text.decode(self.encoding)

        tokens = self.sp_model.EncodeAsPieces(text)
        in_vocab_tokens = []
        for token in tokens:
            if token in self.vocab:
                in_vocab_tokens.append(token)
            else:
                in_vocab_tokens.append(self.unk_token)
        return in_vocab_tokens

    def __call__(self, text, pair_text=None):
        """
        End-to-end tokenization for BigBird models.

        Args:
            text (str): The text to be tokenized.
            pair_text(str):  The pair text to be tokenized.
        Returns:
            `List`: A list of string representing converted tokens.
        """
        return self._tokenize(text)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (list of string) in a single string. Since
        the usage of WordPiece introducing `##` to concat subwords, also remove
        `##` when converting.

        Args:
            tokens (list): A list of string representing tokens to be converted.
        Returns:
            `Str`: Converted string from tokens.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def encode(self,
               text,
               max_seq_len=None,
               max_pred_len=None,
               masked_lm_prob=0.15):
        """
        Returns a tuple containing the encoded sequence and mask information.

        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            max_seq_len (:obj:`int`, `optional`, defaults to`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If set to None, will not limit the total sequence.
            max_pred_len (:obj:`int`, `optional`, defaults to `None`):
                If set to a number, will limit the mask sequence returned so that it has a maximum prediction length.
                If set to None, will not limit the mask sequence.
            masked_lm_prob (:obj:`int`, 'optional`):
                The probability a training data will be masked. Defaults to `0.15`.

        returns:

        Args:
            span_ids()
            masked_lm_positions
            masked_lm_ids
            masked_lm_weights




        """

        def get_input_ids(text):
            if isinstance(text, str):
                text = re.sub('[\n]+', '', text)
                tokens = self._tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        ids = get_input_ids(text)
        # Find the span for in the text 
        max_seq_len = len(ids) if max_seq_len is None else max_seq_len
        max_pred_len = len(ids) if max_pred_len is None else max_pred_len

        end_pos = max_seq_len - 2 + np.random.randint(
            max(1, len(ids) - max_seq_len - 2))
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
        words = np.split(
            np.arange(
                len(span_ids), dtype="int32"), word_begin_pos)[1:]
        assert len(words) == len(word_begin_pos)
        num_to_predict = min(
            max_pred_len,
            max(1, int(round(len(word_begin_pos) * masked_lm_prob))))

        masked_lm_positions = np.concatenate(
            np.random.choice(
                np.array(
                    [[]] + words, dtype=np.object)[1:],
                num_to_predict,
                replace=False),
            0)
        if len(masked_lm_positions) > max_pred_len:
            masked_lm_positions = masked_lm_positions[:max_pred_len + 1]
            truncate_masking_flag = np.flatnonzero(word_begin_flag[
                masked_lm_positions])
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
        span_ids[random_pos] = np.random.randint(
            self.unk_id + 1, self.vocab_size, len(random_pos), dtype=np.int32)
        span_ids = np.concatenate([
            np.array(
                [self.cls_id], dtype=np.int32), span_ids, np.array(
                    [self.sep_id], dtype=np.int32)
        ])
        padding_len = max_seq_len - num_tokens - 2
        span_ids = np.pad(span_ids, [0, padding_len], "constant")
        pred_padding_len = max_pred_len - len(masked_lm_positions)
        masked_lm_weights = np.pad(np.ones_like(
            masked_lm_positions, dtype=np.float32), [0, pred_padding_len],
                                   "constant")
        masked_lm_positions = np.pad(masked_lm_positions + 1,
                                     [0, pred_padding_len], "constant")
        masked_lm_ids = np.pad(masked_lm_ids, [0, pred_padding_len], "constant")
        return span_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.

        Args:
            pair (`bool`): Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False. 

        Returns:
            `Int`: Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. 
        
        A BERT sequence has the following format:

            - single sequence: `[CLS] X [SEP]`
            - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_id] + token_ids_0 + [self.sep_id]
        _cls = [self.cls_id]
        _sep = [self.sep_id]
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep
