# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

"""Tokenization classes for ChatGLM."""
import os
from typing import Dict, List, Optional, Union

import numpy as np
import sentencepiece as spm

from .. import PretrainedTokenizer
from ..tokenizer_utils_base import BatchEncoding, PaddingStrategy


class ChatGLMTokenizer(PretrainedTokenizer):
    """
    Construct a ChatGLM tokenizer.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    resource_files_names = {"vocab_file": "ice_text.model"}
    max_model_input_sizes = {"THUDM/chatglm-6b": 2048, "THUDM/chatglm-6b-v1.1": 2048}
    model_input_names = ["input_ids", "attention_mask"]
    pretrained_resource_files_map = {
        "model_file": {
            "THUDM/chatglm-6b": "https://paddlenlp.bj.bcebos.com/models/community/THUDM/chatglm-6b/ice_text.model",
            "THUDM/chatglm-6b-v1.1": "https://paddlenlp.bj.bcebos.com/models/community/THUDM/chatglm-6b-v1.1/ice_text.model",
        }
    }

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<sop>",
        eos_token="<eop>",
        end_token="</s>",
        mask_token="[MASK]",
        gmask_token="[gMASK]",
        pad_token="<pad>",
        padding_side="left",
        do_lower_case=False,
        num_image_tokens=20000,
        **kwargs
    ) -> None:
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            padding_side=padding_side,
            **kwargs,
        )
        self.end_token = end_token
        self.gmask_token = gmask_token
        self.do_lower_case = do_lower_case
        self.vocab_file = vocab_file
        self.num_image_tokens = num_image_tokens
        self.max_blank_length = kwargs.get("max_blank_length", 80)

        self.sp_tokenizer = spm.SentencePieceProcessor()
        self.sp_tokenizer.Load(self.vocab_file)

    @property
    def gmask_token_id(self) -> Optional[int]:
        if self.gmask_token is None:
            return None
        return self.convert_tokens_to_ids(self.gmask_token)

    @property
    def end_token_id(self) -> Optional[int]:
        if self.end_token is None:
            return None
        return self.convert_tokens_to_ids(self.end_token)

    @property
    def tab_token(self):
        return "<|tab|>"

    @staticmethod
    def get_blank_token(length: int):
        assert length >= 2
        return f"<|blank_{length}|>"

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_tokenizer.vocab_size() + self.num_image_tokens

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        if kwargs.get("remove_space", False):
            text = " ".join(text.strip().split())
        if kwargs.get("linebreak", True):
            text = text.replace("\n", "<n>")
        if kwargs.get("whitespaces", True):
            text = text.replace("\t", self.tab_token)
            for i in range(self.max_blank_length, 1, -1):
                text = text.replace(" " * i, self.get_blank_token(i))
        return (text, kwargs)

    def _tokenize(self, text, **kwargs):
        """Returns a tokenized string."""
        add_dummy_prefix = kwargs.get("add_dummy_prefix", True)

        if not add_dummy_prefix:
            text = "<n>" + text
        tokens = self.sp_tokenizer.EncodeAsPieces(text)
        return tokens if add_dummy_prefix else tokens[2:]

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs
    ) -> str:
        token_ids = [int(_id) - self.num_image_tokens for _id in token_ids]
        token_ids = [_id for _id in token_ids if _id >= 0]
        text = super()._decode(
            token_ids,
            skip_special_tokens,
            clean_up_tokenization_spaces,
            spaces_between_special_tokens,
            **kwargs,
        )
        return self.postprocess(text)

    def postprocess(self, text):
        # Postprocess.
        text = text.replace("<n>", "\n")
        text = text.replace(self.tab_token, "\t")
        for i in range(2, self.max_blank_length + 1):
            text = text.replace(self.get_blank_token(i), " " * i)
        return text

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token.startswith("<image_") and token.endswith(">") and token[7:-1].isdigit():
            return int(token[7:-1])
        else:
            return self.sp_tokenizer.PieceToId(token) + self.num_image_tokens

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index >= self.vocab_size:
            return self.unk_token
        else:
            if index < self.num_image_tokens:
                return "<image_{}>".format(index)
            else:
                return self.sp_tokenizer.IdToPiece(index - self.num_image_tokens)

    def convert_tokens_to_string(self, tokens):
        text = self.sp_tokenizer.DecodePieces(tokens)
        text = self.postprocess(text)
        return text

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

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        token_ids_0 += [self.gmask_token_id, self.bos_token_id]
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1 + [self.eos_token_id]
        return token_ids_0

    def _pad(
        self,
        encoded_inputs: Union[Dict, BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names or "attention_mask" in encoded_inputs

        assert self.padding_side == "left"
        required_input = encoded_inputs[self.model_input_names[0]]
        seq_length = len(required_input)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if max_length is not None:
            if self.bos_token_id in required_input:
                context_length = required_input.index(self.bos_token_id)
            else:
                context_length = seq_length
            if "attention_mask" not in encoded_inputs:
                attention_mask = np.ones((1, seq_length, seq_length))
                attention_mask = np.tril(attention_mask)
                attention_mask[:, :, :context_length] = 1
                encoded_inputs["attention_mask"] = attention_mask

            if "position_ids" not in encoded_inputs:
                position_ids = np.arange(seq_length, dtype=np.int64)
                mask_token = self.mask_token_id if self.mask_token_id in required_input else self.gmask_token_id
                if mask_token in required_input:
                    mask_position = required_input.index(mask_token)
                    position_ids[context_length:] = mask_position
                block_position_ids = np.concatenate(
                    [
                        np.zeros(context_length, dtype=np.int64),
                        np.arange(1, seq_length - context_length + 1, dtype=np.int64),
                    ]
                )
                encoded_inputs["position_ids"] = np.stack([position_ids, block_position_ids], axis=0)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = np.pad(
                    encoded_inputs["attention_mask"],
                    pad_width=[(0, 0), (difference, 0), (difference, 0)],
                    mode="constant",
                    constant_values=0,
                )
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                    "token_type_ids"
                ]
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
            if "position_ids" in encoded_inputs:
                encoded_inputs["position_ids"] = np.pad(
                    encoded_inputs["position_ids"], pad_width=[(0, 0), (difference, 0)]
                )
            encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

        return encoded_inputs
