# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sentencepiece as spm

from ...utils.log import logger
from .. import PretrainedTokenizer
from ..tokenizer_utils_base import (
    AddedToken,
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
)

__all__ = ["GemmaTokenizer"]

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

SPIECE_UNDERLINE = "▁"


class GemmaTokenizer(PretrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]
    resource_files_names = VOCAB_FILES_NAMES
    pretrained_resource_files_map = {
        "vocab_file": {
            "google/gemma-7b": "https://bj.bcebos.com/paddlenlp/models/community/google/gemma-7b/tokenizer.model",
            "google/gemma-2b": "https://bj.bcebos.com/paddlenlp/models/community/google/gemma-2b/tokenizer.model",
        },
    }

    pretrained_init_configuration = {
        "google/gemma-7b": {},
    }

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=False,
        spaces_between_special_tokens=False,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, normalized=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, normalized=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, normalized=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, normalized=False) if isinstance(pad_token, str) else pad_token

        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    # Copied from transformers.models.llama.tokenizer_llama.LlamaTokenizer.__setstate__
    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.vocab_size
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def __len__(self):
        """
        Returns the vocabulary size. added_tokens_encoder has to be added in the sp_model
        """
        added_size = 0

        for id in self.added_tokens_decoder:
            if id >= self.sp_model.get_piece_size():
                added_size += 1

        return self.vocab_size + added_size

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.get_vocab
    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string. The Gemma tokenizer never adds a prefix space.
        """
        return self.sp_model.encode(text, out_type=str)

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self.sp_model.PieceToId(token)

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.added_tokens_decoder:
            return self.added_tokens_decoder[index]
        token = self.sp_model.IdToPiece(index)
        return token

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        sub_texts = []
        current_sub_text = []
        for ids in token_ids:
            if skip_special_tokens and ids in self.all_special_ids:
                continue
            if ids in self.added_tokens_decoder:
                if current_sub_text:
                    sub_texts.append(self.sp_model.decode(current_sub_text))
                cur_id = self.added_tokens_decoder[ids]
                if isinstance(cur_id, AddedToken):
                    sub_texts.append(cur_id.content)
                elif isinstance(cur_id, str):
                    sub_texts.append(cur_id)
                current_sub_text = []
            elif ids in self.all_special_ids:
                if current_sub_text:
                    sub_texts.append(self.sp_model.decode(current_sub_text))
                sub_texts.append(self._convert_id_to_token(ids))
                current_sub_text = []
            else:
                current_sub_text.append(ids)
        if current_sub_text:
            sub_texts.append(self.sp_model.decode(current_sub_text))

        if spaces_between_special_tokens:
            sub_texts = " ".join(sub_texts)
        else:
            sub_texts = "".join(sub_texts)

        return sub_texts

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.added_tokens_encoder:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            elif token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.save_vocabulary
    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        For Zero Padding, Copied from llama

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults

        # attention_mask shape [1,seq_len,seq_len]
        if "attention_mask" in encoded_inputs and len(np.shape(encoded_inputs["attention_mask"])) > 2:
            attention_mask = encoded_inputs["attention_mask"]
            encoded_inputs.pop("attention_mask")
        else:
            attention_mask = None

        required_input = encoded_inputs[self.model_input_names[0]]
        encoded_inputs = super()._pad(
            encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask
        )
        if attention_mask is not None and len(np.shape(attention_mask)) > 2:
            encoded_inputs["attention_mask"] = attention_mask
            needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length
            if needs_to_be_padded:
                difference = max_length - len(required_input)
                if "attention_mask" in encoded_inputs:
                    encoded_inputs["attention_mask"] = np.pad(
                        encoded_inputs["attention_mask"],
                        pad_width=[(0, 0), (difference, 0), (difference, 0)],
                        mode="constant",
                        constant_values=0,
                    )
        return encoded_inputs
