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

from ..ernie.tokenizer import ErnieTokenizer

__all__ = ['ErnieGramTokenizer']


class ErnieGramTokenizer(ErnieTokenizer):
    r"""
    Constructs an ERNIE-Gram tokenizer. It uses a basic tokenizer to do punctuation
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
            from paddlenlp.transformers import ErnieGramTokenizer
            tokenizer = ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
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
            "ernie-gram-zh":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_gram_zh/vocab.txt",
        }
    }
    pretrained_init_configuration = {"ernie-gram-zh": {"do_lower_case": True}, }

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]"):
        super(ErnieGramTokenizer, self).__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token)
