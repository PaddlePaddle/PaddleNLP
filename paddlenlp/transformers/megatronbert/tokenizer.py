# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The HuggingFace Inc. team.
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
"""Tokenization classes for MegatronBert."""

from .. import BertTokenizer

__all__ = ["MegatronBertTokenizer"]

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"megatronbert-cased": 512, "megatronbert-uncased": 512}


class MegatronBertTokenizer(BertTokenizer):
    """
    Constructs a MegatronBert tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.

    Args:
        vocab_file (str):
            The vocabulary file path (ends with '.txt') required to instantiate
            a `WordpieceTokenizer`.
        do_lower_case (bool):
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

    Examples:
        .. code-block::

            from paddlenlp.transformers import MegatronBertTokenizer
            tokenizer = MegatronBertTokenizer.from_pretrained('MegatronBert-uncased')
            inputs = tokenizer('He was a puppeteer')
            print(inputs)

            '''
            {'input_ids': [101, 2002, 2001, 1037, 13997, 11510, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0]}
            '''

    """

    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "megatronbert-uncased": "https://bj.bcebos.com/paddle-hapi/models/bert/bert-base-uncased-vocab.txt",
            "megatronbert-cased": "https://bj.bcebos.com/paddle-hapi/models/bert/bert-base-cased-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "megatronbert-uncased": {"do_lower_case": True},
        "megatronbert-cased": {"do_lower_case": False},
    }
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super(MegatronBertTokenizer, self).__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
