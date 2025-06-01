# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""Tokenization classes for Gemma2."""

from typing import Dict, List, Optional, Tuple, Union

from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.transformers.tokenizer_utils_base import PaddingStrategy, TensorType

__all__ = ["Gemma2Tokenizer"]


class Gemma2Tokenizer(PretrainedTokenizer):
    """
    Construct a Gemma2 tokenizer. Based on SentencePiece.
    """

    resource_files_names = {"vocab_file": "vocab.model", "tokenizer_config_file": "tokenizer_config.json"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "gemma2-2b": "https://bj.bcebos.com/paddlenlp/models/transformers/gemma2/gemma2-2b-vocab.model",
            "gemma2-7b": "https://bj.bcebos.com/paddlenlp/models/transformers/gemma2/gemma2-7b-vocab.model",
        },
        "tokenizer_config_file": {
            "gemma2-2b": "https://bj.bcebos.com/paddlenlp/models/transformers/gemma2/gemma2-2b-tokenizer_config.json",
            "gemma2-7b": "https://bj.bcebos.com/paddlenlp/models/transformers/gemma2/gemma2-7b-tokenizer_config.json",
        },
    }
    pretrained_init_configuration = {
        "gemma2-2b": {"do_lower_case": True},
        "gemma2-7b": {"do_lower_case": True},
    }
    max_model_input_sizes = {
        "gemma2-2b": 8192,
        "gemma2-7b": 8192,
    }
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        tokenizer_config_file: Optional[str] = None,
        do_lower_case: bool = True,
        remove_space: bool = True,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        **kwargs,
    ) -> None:
        """
        Args:
            vocab_file (str): Path to the vocabulary file.
            tokenizer_config_file (str, optional): Path to the tokenizer config file.
            do_lower_case (bool): Whether to lowercase the input when tokenizing.
            remove_space (bool): Whether to remove spaces when tokenizing.
            bos_token (str): The beginning of sequence token.
            eos_token (str): The end of sequence token.
            unk_token (str): The unknown token.
            pad_token (str): The token used for padding.
        """
        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

        import sentencepiece as spm
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        self.vocab_file = vocab_file

        # Initialize token-to-id and id-to-token mappings
        self.encoder: Dict[str, int] = {self.sp_model.id_to_piece(i): i for i in range(self.sp_model.get_piece_size())}
        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}

        # Special token mappings
        self.special_tokens_encoder: Dict[str, int] = {}
        self.special_tokens_decoder: Dict[int, str] = {}
        for token in [self.bos_token, self.eos_token, self.unk_token, self.pad_token]:
            if token not in self.encoder:
                continue
            index = self.encoder[token]
            self.special_tokens_encoder[token] = index
            self.special_tokens_decoder[index] = token

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dictionary of token to index."""
        return self.encoder

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string using SentencePiece."""
        if self.do_lower_case:
            text = text.lower()
        if self.remove_space:
            text = "".join(text.split())

        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token to an id using the vocab."""
        if token in self.special_tokens_encoder:
            return self.special_tokens_encoder[token]
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocab."""
        if index in self.special_tokens_decoder:
            return self.special_tokens_decoder[index]
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens to a single string."""
        return "".join(tokens).replace("▁", " ").strip()

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed.
        """
        sep = [self.eos_token_id]
        cls = [self.bos_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        """Save the vocabulary and special tokens file to a directory."""
        import os
        import shutil

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        if filename_prefix is None:
            filename_prefix = ""

        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + self.resource_files_names["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            shutil.copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
