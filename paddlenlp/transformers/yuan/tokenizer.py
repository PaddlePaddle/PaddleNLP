# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

"""Tokenization class for Yuan2.0 model"""

import os
import re
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from ...utils.log import logger
from .. import PretrainedTokenizer

__all__ = ["YuanTokenizer"]


class YuanTokenizer(PretrainedTokenizer):
    """
    YuanTokenizer is equivalent to LlamaTokenizer
    """

    model_input_names = ["input_ids", "attention_mask", "position_ids"]
    resource_files_names = {
        "vocab_file": "tokenizer.model",
    }

    padding_side = "left"

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_bos_token=True,
        add_eos_token=False,
        sp_model_kwargs=None,
        decode_with_prefix_space=False,
        **kwargs
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)

        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.decode_with_prefix_space = decode_with_prefix_space
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        self.pad_token_id = self.eos_token_id

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.sp_model.bos_id()

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.sp_model.eos_id()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Returns a tokenized string."""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

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
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + self.resource_files_names["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is not None:
            output = output + token_ids_1

        if self.add_eos_token:
            output = output + [self.eos_token_id]

        return output

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

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def _encode_chat_inputs(
        self,
        conversations: List[Tuple[str, str]],
        context_data: Dict[str, Any] = {},
        system: str = None,
        add_generation_prompt=True,
    ):
        result = {}

        # Some template do not support system msg, so we need to check it first.
        if system:
            try:
                self.chat_template.render(messages={"role": "system", "content": system})
            except Exception as e:
                raise ValueError("System is not supported in this tokenizer.", e)

        # convert list msg to role dict msg
        conversation_dict = []
        origin_msg = []
        for round in conversations:
            round_role = [
                {"role": "user", "content": round[0]},
                {"role": "assistant", "content": round[1]},
            ]
            origin_msg.extend(round_role)
            conversation_dict.append(round_role)
        ans = []

        # get answer in single round, then compile the chat entirely and split by single round ans
        # attention: answer should include end token!
        for conv in conversation_dict:
            roundi = [system] + conv if system else conv
            roundi_str = self.chat_template.render(
                messages=roundi, add_generation_prompt=False, **self.special_tokens_map
            )
            roundi_no_ans = [system] + [conv[0]] if system else [conv[0]]
            roundi_no_ans_str = self.chat_template.render(
                messages=roundi_no_ans, add_generation_prompt=add_generation_prompt, **self.special_tokens_map
            )

            ans_roundi = roundi_str[len(roundi_no_ans_str) - len("<sep>") + len("<n>") : -len("<sep>")]
            ans.append(ans_roundi)
        for idx, _ in enumerate(ans):
            ans[idx] += "<n>" if idx != len(ans) - 1 else "<sep>"

        non_learnable_parts = self._extract_non_learnable_parts(origin_msg, ans)
        assert len(non_learnable_parts) == len(ans)

        conversation_ids = []
        for i in range(len(non_learnable_parts)):
            conversation_ids.append(
                self.batch_encode(
                    [non_learnable_parts[i], ans[i]],
                    add_special_tokens=False,
                    padding=False,
                )["input_ids"]
            )

        result["conversations"] = conversation_ids
        return result

    def _extract_non_learnable_parts(self, origin_msg: List[Dict[str, str]], split_s: List[str]):
        """Split the entire chat by specified words. Extract the non-learnable parts."""
        # distingish and replace the special words in original string to an uncompiled form: Like | -> \|
        split_s_with_front_token = split_s.copy()
        for idx, _ in enumerate(split_s):
            split_s_with_front_token[idx] = "<n>" + split_s_with_front_token[idx]
        regex_pattern = "|".join(map(re.escape, split_s_with_front_token))
        # splited by replaced specified words
        non_learnable_parts = re.split(
            r"(?:%s)" % regex_pattern,
            self.chat_template.render(messages=origin_msg, add_generation_prompt=False, **self.special_tokens_map),
        )
        if non_learnable_parts[-1] == "":
            non_learnable_parts.pop()
        for idx, _ in enumerate(non_learnable_parts):
            non_learnable_parts[idx] = non_learnable_parts[idx] + "<n>"
        return non_learnable_parts
