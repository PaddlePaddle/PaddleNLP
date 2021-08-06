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
    Constructs an ERNIE-Doc tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.

    Args:
        vocab_file (str): 
            file path of the vocabulary.
        do_lower_case (str, optional): 
            Whether the text strips accents and convert to lower case. 
            Defaults to `True`.
        unk_token (str, optional): 
            The special token for unknown words. 
            Defaults to "[UNK]".
        sep_token (str, optional): 
            The special token for separator token. 
            Defaults to "[SEP]".
        pad_token (str, optional): 
            The special token for padding. 
            Defaults to "[PAD]".
        cls_token (str, optional): 
            The special token for cls. 
            Defaults to "[CLS]".
        mask_token (str, optional): 
            The special token for mask.
            Defaults to "[MASK]".
    
    Examples:
        .. code-block:: python
            from paddlenlp.transformers import ErnieDocTokenizer
            tokenizer = ErnieDocTokenizer.from_pretrained('ernie-doc-base-zh')
            encoded_inputs = tokenizer('这是一个测试样例')
            # encoded_inputs: 
            # { 
            #   'input_ids': [1, 47, 10, 7, 27, 558, 525, 314, 656, 2], 
            #   'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # }


    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-doc-base-zh":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie-doc-base-zh/vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "ernie-doc-base-zh": {
            "do_lower_case": True
        },
    }

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]"):
        super(ErnieDocTokenizer, self).__init__(
            vocab_file,
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

    Args:
        vocab_file (str): 
            file path of the vocabulary.
        encoder_json_path (str, optional):
            file path of the id to vocab.
        vocab_bpe_path (str, optional):
            file path of word merge text.
        unk_token (str, optional): 
            The special token for unknown words. 
            Defaults to "[UNK]".
        sep_token (str, optional): 
            The special token for separator token. 
            Defaults to "[SEP]".
        pad_token (str, optional): 
            The special token for padding. 
            Defaults to "[PAD]".
        cls_token (str, optional): 
            The special token for cls. 
            Defaults to "[CLS]".
        mask_token (str, optional): 
            The special token for mask.
            Defaults to "[MASK]".
    
    Examples:
        .. code-block:: python
            from paddlenlp.transformers import ErnieDocBPETokenizer
            tokenizer = ErnieDocBPETokenizer.from_pretrained('ernie-doc-base-en')
            encoded_inputs = tokenizer('This is an example')
            # encoded_inputs: 
            # { 
            #   'input_ids': [713, 16, 41, 1246], 
            #   'token_type_ids': [0, 0, 0, 0]
            # }


    """
    resource_files_names = {
        "vocab_file": "vocab.txt",
        "encoder_json_path": "encoder.json",
        "vocab_bpe_path": "vocab.bpe"
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-doc-base-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie-doc-base-en/vocab.txt"
        },
        "encoder_json_path": {
            "ernie-doc-base-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie-doc-base-en/encoder.json"
        },
        "vocab_bpe_path": {
            "ernie-doc-base-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie-doc-base-en/vocab.bpe"
        }
    }
    pretrained_init_configuration = {
        "ernie-doc-base-en": {
            "unk_token": "[UNK]"
        },
    }

    def __init__(self,
                 vocab_file,
                 encoder_json_path="./configs/encoder.json",
                 vocab_bpe_path="./configs/vocab.bpe",
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]"):
        super(ErnieDocBPETokenizer, self).__init__(
            vocab_file,
            encoder_json_path=encoder_json_path,
            vocab_bpe_path=vocab_bpe_path,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token)
