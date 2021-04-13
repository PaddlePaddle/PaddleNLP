# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.

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
import pickle
import six
import shutil

from paddle.utils import try_import
from paddlenlp.utils.env import MODEL_HOME

from .. import BasicTokenizer, PretrainedTokenizer

__all__ = ['ErnieCtmTokenizer']


class ErnieCtmTokenizer(PretrainedTokenizer):
    r"""
    Construct a ERNIE-CTM tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.
    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to do basic tokenization before WordPiece.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token_template (:obj:`str`, `optional` defauts to :obj:`"[CLS{}]"`)
            The template of summary token for multiple summary placeholders.
        summary_num (:obj:`int`, `optional`, defaults to 1):
            Summary placeholder used in ernie-ctm model. For catching a sentence global feature from multiple aware.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-ctm":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_ctm/vocab.txt"
        }
    }
    pretrained_init_configuration = {"ernie-ctm": {"do_lower_case": True}}

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token_template="[CLS{}]",
                 summary_num=1,
                 mask_token="[MASK]",
                 **kwargs):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = ErnieTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.cls_token_template = cls_token_template
        self.summary_num = summary_num
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Build model inputs from a sequence or a pair of sequences for sequence classification tasks by
        concatenating and add special tokens. A ERNIE-CTM sequence has the following format:

        - single sequence: [CLS0][CLS1]... X [SEP]
        - pair of sequences: [CLS0][CLS1]... X [SEP] X [SEP]

        Arguments:
            token_ids_0 {typing.List[int]} -- List of IDs to which the special tokens will be added.

        Keyword Arguments:
            token_ids_1 {typing.Optional[typing.List[int]]} -- Optional second list of IDs for sequence pairs.
            (default: {None})

        Returns:
            typing.List[int] -- List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        cls_token_ids = [
            self.convert_tokens_to_ids(self.cls_token_template.format(sid))
            for sid in range(self.summary_num)
        ]
        if token_ids_1 is None:
            return cls_token_ids + token_ids_0 + [self.sep_token_id]
        return cls_token_ids + token_ids_0 + [
            self.sep_token_id
        ] + token_ids_1 + [self.sep_token_id]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """Retrieve sequence ids from a token list that has no special tokens added. This method is called when
        adding special tokens using the tokenizer ``prepare_for_model`` method.

        Arguments:
            token_ids_0 {typing.List[int]} -- List of IDs.

        Keyword Arguments:
            token_ids_1 {typing.Optional[typing.List[int]]} --
                Optional seconde list of IDs for sequence pairs. (default: {None})
            already_has_special_tokens {bool} --
                Whether or not the token list is already formatted with special tokens for the model. (default: {False})

        Returns:
            typing.List[int] -- A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
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
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
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
        if token_ids_1 is None:
            return (self.summary_num + len(token_ids_0 + sep)) * [0]
        return (self.summary_num + len(token_ids_0 + sep)
                ) * [0] + len(token_ids_1 + sep) * [1]

    def num_special_tokens_to_add(self, pair=False):
        if pair is True:
            return self.summary_num + 2
        else:
            return self.summary_num + 1

    def tokenize(self, text, **kwargs):
        """
        Basic Tokenization of a piece of text, to tokenize Chinese Character, we should transform string to token list
        straightly.
        """
        orig_tokens = list(text)
        output_tokens = []
        for token in orig_tokens:
            if self.do_lower_case is True:
                token = token.lower()
            output_tokens.append(token)
        return output_tokens
