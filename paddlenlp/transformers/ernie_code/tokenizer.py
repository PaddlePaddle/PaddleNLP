# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 Baidu ErnieCode Authors.
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

import unicodedata
from collections import UserDict
from typing import List, Union

import numpy as np
import paddle

from ..t5.tokenizer import T5Tokenizer

__all__ = [
    "ErnieCodeTokenizer",
]

formate_dict = {" ": "<|space|>"}


def to_py_obj(obj):
    """
    Convert a Paddle tensor, Numpy array or python list to a python list.
    """
    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    elif isinstance(obj, paddle.Tensor):
        return obj.numpy().tolist()
    elif isinstance(obj, (np.ndarray, np.number)):  # tolist also works on 0d np arrays
        return obj.tolist()
    else:
        return obj


def clean_up_codem_spaces(s: str):
    # post process
    # ===========================
    new_tokens = ["<pad>", "</s>", "<unk>", "\n", "\t", "<|space|>" * 4, "<|space|>" * 2, "<|space|>"]
    for tok in new_tokens:
        s = s.replace(f"{tok} ", tok)

    cleaned_tokens = ["<pad>", "</s>", "<unk>"]
    for tok in cleaned_tokens:
        s = s.replace(tok, "")
    s = s.replace("<|space|>", " ")
    # ===========================
    return s


class ErnieCodeTokenizer(T5Tokenizer):
    """
    Constructs a ErnieCode tokenizer based on SentencePiece .
    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        sentencepiece_model_file (str):
            The vocabulary file (ends with '.spm') required to instantiate
            a `SentencePiece <https://github.com/google/sentencepiece>`__ tokenizer.
        do_lower_case (bool):
            Whether or not to lowercase the input when tokenizing. Defaults to `False`.
        remove_space (bool):
            Whether or note to remove space when tokenizing. Defaults to `True`.
        keep_accents (bool):
            Whether or note to keep accents when tokenizing. Defaults to `False`.
        eos_token (str):
            A special token representing the *eos (end-of-sentence)* token.
            Defaults to "</s>".
        unk_token (str):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "<unk>".
        pad_token (str):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "<pad>".

    """

    resource_files_names = {"sentencepiece_model_file": "spiece.model"}
    pretrained_resource_files_map = {
        "sentencepiece_model_file": {
            "ernie-code-base": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-code/ernie-code-base/spiece.model",
            "ernie-code-base-L512": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-code/ernie-code-base-L512/spiece.model",
        },
    }

    pretrained_init_configuration = {
        "ernie-code-base": {"do_lower_case": False},
        "ernie-code-base-L512": {"do_lower_case": False},
    }

    def __init__(
        self,
        sentencepiece_model_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=True,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=0,
        additional_special_tokens=[],
        sp_model_kwargs=None,
        **kwargs
    ):
        if additional_special_tokens is None or 0 == len(additional_special_tokens):
            additional_special_tokens = [
                "\n",
                "\t",
                "<|space|><|space|><|space|><|space|>",
                "<|space|><|space|>",
                "<|space|>",
            ]

        super(ErnieCodeTokenizer, self).__init__(
            sentencepiece_model_file,
            do_lower_case,
            remove_space,
            keep_accents,
            eos_token,
            unk_token,
            pad_token,
            extra_ids,
            additional_special_tokens,
            sp_model_kwargs,
            **kwargs,
        )

    def preprocess_text(self, inputs: str):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        tokens = list(outputs)
        i = 0
        while i < len(tokens):
            if "\n" == outputs[i]:

                while i + 1 < len(tokens) and " " == tokens[i + 1]:
                    tokens[i + 1] = formate_dict.get(" ")
                    i += 1
            i += 1
        formatted_line = "".join(tokens)
        return formatted_line

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "paddle.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.
        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.
        Args:
            token_ids (`Union[int, List[int], np.ndarray, paddle.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the tokenization spaces.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.
        Returns:
            `str`: The decoded sentence.
        """
        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)

        decoded_preds = self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        return clean_up_codem_spaces(decoded_preds)
