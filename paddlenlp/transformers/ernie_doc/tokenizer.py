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
import os
import pickle
import shutil
import json

from paddlenlp.utils.env import MODEL_HOME
from .. import PretrainedTokenizer, BPETokenizer
from ..ernie.tokenizer import ErnieTokenizer

__all__ = ['ErnieDocTokenizer', 'ErnieDocBPETokenizer']


class ErnieDocTokenizer(ErnieTokenizer):
    r"""
    Constructs an ERNIE-Doc tokenizer.
    It uses a basic tokenizer to do punctuation splitting, lower casing and so on,
    and follows a WordPiece tokenizer to tokenize as subwords.

    This tokenizer inherits from :class:`~paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer`.
    For more information regarding those methods, please refer to this superclass.

    Args:
        vocab_file (str):
            The vocabulary file path (ends with '.txt') required to instantiate
            a `WordpieceTokenizer`.
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
    
    Examples:
        .. code-block::

            from paddlenlp.transformers import ErnieDocTokenizer
            tokenizer = ErnieDocTokenizer.from_pretrained('ernie-doc-base-zh')
            encoded_inputs = tokenizer('He was a puppeteer')

    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-doc-base-zh":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-doc-base-zh/vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "ernie-doc-base-zh": {
            "do_lower_case": True
        },
    }

    max_model_input_sizes = {"ernie-doc-base-en": 1152}

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 **kwargs):
        super(ErnieDocTokenizer, self).__init__(vocab_file,
                                                do_lower_case=do_lower_case,
                                                unk_token=unk_token,
                                                sep_token=sep_token,
                                                pad_token=pad_token,
                                                cls_token=cls_token,
                                                mask_token=mask_token)


class ErnieDocBPETokenizer(BPETokenizer):
    r"""
    Constructs an ERNIE-Doc BPE tokenizer. It uses a bpe tokenizer to do punctuation
    splitting, lower casing and so on, then tokenize words as subwords.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.BPETokenizer`.
    For more information regarding those methods, please refer to this superclass.

    Args:
        vocab_file (str): 
            File path of the vocabulary.
        encoder_json_path (str, optional):
            File path of the id to vocab.
        vocab_bpe_path (str, optional):
            File path of word merge text.
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
    
    Examples:
        .. code-block::

            from paddlenlp.transformers import ErnieDocBPETokenizer
            tokenizer = ErnieDocBPETokenizer.from_pretrained('ernie-doc-base-en')
            encoded_inputs = tokenizer('He was a puppeteer')

    """
    resource_files_names = {
        "vocab_file": "vocab.txt",
        "encoder_json_path": "encoder.json",
        "vocab_bpe_path": "vocab.bpe"
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-doc-base-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-doc-base-en/vocab.txt"
        },
        "encoder_json_path": {
            "ernie-doc-base-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-doc-base-en/encoder.json"
        },
        "vocab_bpe_path": {
            "ernie-doc-base-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-doc-base-en/vocab.bpe"
        }
    }
    pretrained_init_configuration = {
        "ernie-doc-base-en": {
            "unk_token": "[UNK]"
        },
    }
    max_model_input_sizes = {"ernie-doc-base-en": 1152}

    def __init__(self,
                 vocab_file,
                 encoder_json_path="./configs/encoder.json",
                 vocab_bpe_path="./configs/vocab.bpe",
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 **kwargs):
        super(ErnieDocBPETokenizer,
              self).__init__(vocab_file,
                             encoder_json_path=encoder_json_path,
                             vocab_bpe_path=vocab_bpe_path,
                             unk_token=unk_token,
                             sep_token=sep_token,
                             pad_token=pad_token,
                             cls_token=cls_token,
                             mask_token=mask_token)

    @property
    def vocab_size(self):
        """
        Return the size of vocabulary.

        Returns:
            int: The size of vocabulary.
        """
        return len(self.vocab)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. 
        
        A BERT sequence has the following format:

        - single sequence:      ``[CLS] X [SEP]``
        - pair of sequences:        ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs. Defaults to None.

        Returns:
            List[int]: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        """
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens.

        A BERT offset_mapping has the following format:

        - single sequence:      ``(0,0) X (0,0)``
        - pair of sequences:        ``(0,0) A (0,0) B (0,0)``

        Args:
            offset_mapping_ids_0 (List[tuple]):
                List of wordpiece offsets to which the special tokens will be added.
            offset_mapping_ids_1 (List[tuple], optional):
                Optional second list of wordpiece offsets for offset mapping pairs. Defaults to None.

        Returns:
            List[tuple]: A list of wordpiece offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. 

        A BERT sequence pair mask has the following format:
        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (List[int]):
                A list of `inputs_ids` for the first sequence.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs. Defaults to None.

        Returns:
            List[int]: List of token_type_id according to the given sequence(s).
        """
        _sep = [self.sep_token_id]
        _cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(_cls + token_ids_0 + _sep) * [0]
        return len(_cls + token_ids_0 + _sep) * [0] + len(token_ids_1 +
                                                          _sep) * [1]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.

        Args:
            token_ids_0 (List[int]):
                A list of `inputs_ids` for the first sequence.
            token_ids_1 (List[int], optinal):
                Optional second list of IDs for sequence pairs. Defaults to None.
            already_has_special_tokens (bool, optional): Whether or not the token list is already 
                formatted with special tokens for the model. Defaults to None.

        Returns:
            List[int]: The list of integers either be 0 or 1: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(
                    lambda x: 1
                    if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
