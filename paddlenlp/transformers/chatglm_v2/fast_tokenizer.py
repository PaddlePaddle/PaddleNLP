# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from ..tokenizer_utils_fast import PretrainedFastTokenizer
from .tokenizer import ChatGLMv2Tokenizer

VOCAB_FILES_NAMES = {
    # "sentencepiece_model_file": "sentencepiece.bpe.model",
    "vocab_file": "tokenizer.model",
    "tokenizer_file": "tokenizer.json",
}


class ChatGLMv2FastTokenizer(PretrainedFastTokenizer):
    resource_files_names = VOCAB_FILES_NAMES  # for save_pretrained
    slow_tokenizer_class = ChatGLMv2Tokenizer
    pretrained_resource_files_map = slow_tokenizer_class.pretrained_resource_files_map
    pretrained_init_configuration = slow_tokenizer_class.pretrained_init_configuration
    model_input_names = ["input_ids", "attention_mask", "position_ids"]
    padding_side = "left"

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.pad_token_id = 0  # hard code alter

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, self.vocab_files_names["vocab_file"])
        else:
            vocab_file = save_directory

        with open(self.vocab_file, "rb") as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)
