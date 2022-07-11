# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright Google AI and The HuggingFace Inc. team. All rights reserved.
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

from typing import Dict, List, Optional
from .. import PretrainedTokenizer

__all__ = ['CanineTokenizer']

# Unicode defines 1,114,112 total “codepoints”
UNICODE_VOCAB_SIZE = 1114112

# Below: Constants defining canonical codepoints for special, pseudo-characters.
# Copied from https://github.com/google-research/language/blob/master/language/canine/special_codepoints.py

# Padding is always index zero. This means that the NULL character is
# technically not embeddable. This seems fine according to all reasonable
# interpretations of the NULL character as a past-end-of-string marker.
PAD = 0

CLS = 0xE000
SEP = 0xE001
BOS = 0xE002
MASK = 0xE003
RESERVED = 0xE004

# Maps special codepoints to human-readable names.
SPECIAL_CODEPOINTS: Dict[int, str] = {
    # Special symbols are represented using codepoints values that are valid,
    # but designated as "Private Use", meaning that they will never be assigned
    # characters by the Unicode Consortium, and are thus safe for use here.
    #
    # NOTE: Do *NOT* add any sort of [UNK_CHAR] here. They are explicitly
    # excluded and should fail with a hard error.
    CLS: "[CLS]",
    SEP: "[SEP]",
    BOS: "[BOS]",
    MASK: "[MASK]",
    PAD: "[PAD]",
    RESERVED: "[RESERVED]",
}

# Maps special codepoint human-readable names to their codepoint values.
SPECIAL_CODEPOINTS_BY_NAME: Dict[str, int] = {
    name: codepoint
    for codepoint, name in SPECIAL_CODEPOINTS.items()
}


class CanineTokenizer(PretrainedTokenizer):
    r"""
    Construct a Canine tokenizer, which convert text inputs into code points based on
    the characters Unicode code point.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        bos_token (str): The special token for beginning of sequence token. Default: "`chr(0xE002)`".
        eos_token (str): The special token for end of sequence token. Default: "`chr(0xE001)`".
        cls_token (str): The special token for cls. Default: "`chr(0xE000)`".
        sep_token (str): The special token for separator token . Default: "`chr(0xE001)`".
        pad_token (str): The special token for padding. Default: "`chr(0)`".
        pad_token (str): The special token for mask token. Default: "`chr(0xE003)`".
        model_max_length (int): The specified maximum sequence length. Default: "2048".

    Examples:
        .. code-block:: python
            from paddlenlp.transformers import CanineTokenizer

            tokenizer = CanineTokenizer.from_pretrained("canine-s")
            text = "Life is like a box of chocolates."
            inputs = tokenizer(text, padding="longest", truncation=True)

            # above line outputs:
            # {'input_ids': [57344, 76, 105, 102, 101, 32, 105, 115, 32, 108, 105, 107, 101, 32, 97,
            32, 98, 111, 120, 32, 111, 102, 32, 99, 104, 111, 99, 111, 108, 97, 116, 101, 115, 46, 57345],
            'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    """
    pretrained_init_configuration = {
        "canine-s": {
            "model_max_length": 2048
        },
    }

    def __init__(self,
                 bos_token=chr(CLS),
                 pad_token=chr(PAD),
                 eos_token=chr(SEP),
                 cls_token=chr(CLS),
                 sep_token=chr(SEP),
                 mask_token=chr(MASK),
                 model_max_length=2048,
                 **kwargs):
        super().__init__(**kwargs)
        self._special_codepoints: Dict[str, int] = {}
        for codepoint, name in SPECIAL_CODEPOINTS.items():
            self._special_codepoints[name] = codepoint

        self._unicode_vocab_size = UNICODE_VOCAB_SIZE
        self._num_special_tokens = len(self._special_codepoints)

    @property
    def vocab_size(self):
        """
        Return the size of vocabulary.
        Returns:
            int: The size of vocabulary.
        """
        return self._unicode_vocab_size

    def _tokenize(self, text, **kwargs):
        r"""
        Tokenization for Canine models, which is simple character splitting.
        Args:
            text (str): The text to be tokenized.

        Returns:
            List[str]: A list of string representing converted tokens.
        """
        return list(text)

    def _convert_token_to_id(self, token):
        try:
            return ord(token)
        except TypeError:
            raise ValueError(f"invalid token: '{token}'")

    def _convert_id_to_token(self, index):
        try:
            if index in SPECIAL_CODEPOINTS:
                return SPECIAL_CODEPOINTS[index]
            return chr(index)
        except TypeError:
            raise ValueError(f"invalid id: {index}")

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (list of string) to a single string by
        using ``' '.join(tokens)`` .

        Args:
            tokens (list[str]): A sequence of tokens.

        Returns:
            str: Converted string.
        """
        return "".join(tokens)

    def tokenize(self, text, **kwargs):
        """ End-to-end tokenization for Canine models. """
        return self._tokenize(text)

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        This implementation does not add special tokens and this method should be overridden in a subclass.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True)
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] *
                                                       len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids)

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The token type ids.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return [0] * len(cls + token_ids_0 + sep) + [1] * len(token_ids_1 + sep)

    # Copied from paddlenlp.transformers.bert.tokenizer.BertTokenizer.num_special_tokens_to_add
    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Args:
            pair(bool):
                Whether the input is a sequence pair or a single sequence.
                Defaults to `False` and the input is a single sequence.

        Returns:
            int: Number of tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(
                token_ids_0, token_ids_1 if pair else None))

    # Copied from paddlenlp.transformers.bert.tokenizer.BertTokenizer.build_offset_mapping_with_special_tokens
    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        """
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens.

        A BERT offset_mapping has the following format:

        - single sequence:      ``(0,0) X (0,0)``
        - pair of sequences:        ``(0,0) A (0,0) B (0,0)``

        Args:
            offset_mapping_0 (List[tuple]):
                List of wordpiece offsets to which the special tokens will be added.
            offset_mapping_1 (List[tuple], optional):
                Optional second list of wordpiece offsets for offset mapping pairs. Defaults to None.

        Returns:
            List[tuple]: A list of wordpiece offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    @staticmethod
    def save_vocabulary(*args, **kwargs):
        """
        CanineTokenizer has no vocab file.
        """
        return ()

    @property
    def unk_token(self):
        """
        Unk_token is not used for Canine model.
        """
        # logger.info("unk_token is set to be None for Canine model")
        return None

    def get_vocab(self):
        """
        Canine Model has no vocabulary.
        """
        return

    def get_offset_mapping(self, text):
        """
            Returns the map of tokens and the start and end index of their start and end character.

            Args:
                text (str):
                    Input text.
            Returns:
                list: The offset map of input text.

            """
        if text is None:
            return None

        token_mapping = [(idx, idx + 1) for idx in range(len(text))]
        return token_mapping
