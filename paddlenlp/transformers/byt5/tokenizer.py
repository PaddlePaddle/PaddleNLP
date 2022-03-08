# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.utils import try_import
from typing import Dict, List, Optional, Tuple
from ..albert.tokenizer import AlbertEnglishTokenizer
import warnings
from ..utils import InitTrackerMeta, fn_args_to_dict

__all__ = ['ByT5Tokenizer', ]


class ByT5Tokenizer(AlbertEnglishTokenizer):
    """
    Constructs a T5 tokenizer based on SentencePiece .
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

    def __init__(self,
                 do_lower_case=False,
                 remove_space=True,
                 keep_accents=False,
                 eos_token="</s>",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 extra_ids = 125,
                 additional_special_tokens = None,
                 **kwargs):

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self._extra_ids = extra_ids
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.added_tokens_encoder = {"<pad>": 0, "</s>": 1, "<unk>": 2}
        self.added_tokens_decoder = {0: "<pad>", 1: "</s>", 2: "<unk>"}

        self._utf_vocab_size = 2 ** 8  # utf is 8 bits

        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in str(x)), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to ByT5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )
        # define special tokens dict
        self.special_tokens_map = {"eos_token": {"content": "</s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True}, "unk_token": {"content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True}, "pad_token": {"content": "<pad>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True}, "additional_special_tokens": ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "<extra_id_3>", "<extra_id_4>", "<extra_id_5>", "<extra_id_6>", "<extra_id_7>", "<extra_id_8>", "<extra_id_9>", "<extra_id_10>", "<extra_id_11>", "<extra_id_12>", "<extra_id_13>", "<extra_id_14>", "<extra_id_15>", "<extra_id_16>", "<extra_id_17>", "<extra_id_18>", "<extra_id_19>", "<extra_id_20>", "<extra_id_21>", "<extra_id_22>", "<extra_id_23>", "<extra_id_24>", "<extra_id_25>", "<extra_id_26>", "<extra_id_27>", "<extra_id_28>", "<extra_id_29>", "<extra_id_30>", "<extra_id_31>", "<extra_id_32>", "<extra_id_33>", "<extra_id_34>", "<extra_id_35>", "<extra_id_36>", "<extra_id_37>", "<extra_id_38>", "<extra_id_39>", "<extra_id_40>", "<extra_id_41>", "<extra_id_42>", "<extra_id_43>", "<extra_id_44>", "<extra_id_45>", "<extra_id_46>", "<extra_id_47>", "<extra_id_48>", "<extra_id_49>", "<extra_id_50>", "<extra_id_51>", "<extra_id_52>", "<extra_id_53>", "<extra_id_54>", "<extra_id_55>", "<extra_id_56>", "<extra_id_57>", "<extra_id_58>", "<extra_id_59>", "<extra_id_60>", "<extra_id_61>", "<extra_id_62>", "<extra_id_63>", "<extra_id_64>", "<extra_id_65>", "<extra_id_66>", "<extra_id_67>", "<extra_id_68>", "<extra_id_69>", "<extra_id_70>", "<extra_id_71>", "<extra_id_72>", "<extra_id_73>", "<extra_id_74>", "<extra_id_75>", "<extra_id_76>", "<extra_id_77>", "<extra_id_78>", "<extra_id_79>", "<extra_id_80>", "<extra_id_81>", "<extra_id_82>", "<extra_id_83>", "<extra_id_84>", "<extra_id_85>", "<extra_id_86>", "<extra_id_87>", "<extra_id_88>", "<extra_id_89>", "<extra_id_90>", "<extra_id_91>", "<extra_id_92>", "<extra_id_93>", "<extra_id_94>", "<extra_id_95>", "<extra_id_96>", "<extra_id_97>", "<extra_id_98>", "<extra_id_99>", "<extra_id_100>", "<extra_id_101>", "<extra_id_102>", "<extra_id_103>", "<extra_id_104>", "<extra_id_105>", "<extra_id_106>", "<extra_id_107>", "<extra_id_108>", "<extra_id_109>", "<extra_id_110>", "<extra_id_111>", "<extra_id_112>", "<extra_id_113>", "<extra_id_114>", "<extra_id_115>", "<extra_id_116>", "<extra_id_117>", "<extra_id_118>", "<extra_id_119>", "<extra_id_120>", "<extra_id_121>", "<extra_id_122>", "<extra_id_123>", "<extra_id_124>"]}
        self.special_tokens_encoder = {
            self.pad_token: 0,
            self.eos_token: 1,
            self.unk_token: 2,
        }
        self._num_special_tokens = len(self.special_tokens_encoder)
        n = len(additional_special_tokens)
        for i, token in enumerate(additional_special_tokens):
            self.special_tokens_encoder[token] = self.vocab_size + i - n
        self.special_tokens_decoder: Dict[str, int] = {v: k for k, v in self.special_tokens_encoder.items()}
        
    @property
    def vocab_size(self):
        return self._utf_vocab_size + self._num_special_tokens + self._extra_ids
    
    def _wrap_init(self, original_init, *args, **kwargs):
        """
        It would be hooked after `__init__` to add specials tokens (arguments of
        `__init__` whose name ends with `_token`) as attributes of the tokenizer
        instance.
        """
        # expose tokens as attributes
        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        assert self.padding_side in [
            "right", "left"
        ], "Padding side must be either left or right"

        init_dict = fn_args_to_dict(original_init, *args, **kwargs)
        # TODO(guosheng): Use OrderedDict, otherwise `all_special_tokens` returns
        # a list without order.
        special_tokens_map = {}
        for identifier, token in init_dict.items():
            if identifier.endswith('_token'):
                # setattr(self, identifier, token)
                special_tokens_map[identifier] = token
        self.special_tokens_map = special_tokens_map

    def __call__(self,
                 text,
                 text_pair=None,
                 max_seq_len=None,
                 stride=0,
                 is_split_into_words=False,
                 pad_to_max_seq_len=False,
                 truncation_strategy="longest_first",
                 return_position_ids=False,
                 return_token_type_ids=False,
                 return_attention_mask=True,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        return super(ByT5Tokenizer, self).__call__(
            text, text_pair, max_seq_len, stride, is_split_into_words,
            pad_to_max_seq_len, truncation_strategy, return_position_ids,
            return_token_type_ids, return_attention_mask, return_length,
            return_overflowing_tokens, return_special_tokens_mask)

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the end of sentence token in the vocabulary. Returns :obj:`None` if the token has
        not been set.
        """
        if self.eos_token is None:
            return None
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the end of sentence token in the vocabulary. Returns :obj:`None` if the token has
        not been set.
        """
        if self.unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the end of sentence token in the vocabulary. Returns :obj:`None` if the token has
        not been set.
        """
        if self.pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

# 这个函数和t5没有什么区别
    def _add_eos_if_not_present(self, token_ids):
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1):
        """
        Build model inputs from a sequence or a pair of sequence.
        An Reformer sequence has the following format:
        - single sequence:      ``X </s>``
        - pair of sequences:        ``A </s> B </s>``
        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs. Defaults to None.
        Returns:
            List[int]: List of input_id with the appropriate special tokens.
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

#这个与T5也一样
    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences.
        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (List[int]):
                List of IDs.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.
        Returns:
            List[int]: List of token_type_id according to the given sequence(s).
            
        """
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

#这个与T5也一样
    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.
        Args:
            token_ids_0 (List[int]): List of ids of the first sequence.
            token_ids_1 (List[int], optional): List of ids of the second sequence.
            already_has_special_tokens (bool, optional): Whether or not the token list is already
                formatted with special tokens for the model. Defaults to None.
        Returns:
            List[int]: The list of integers in the range [0, 1]:
                1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True, )

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

#重写基类的tokenize方法
    def _tokenize(self, text, sample=False):
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def tokenize(self, text):
        return self._tokenize(text)


    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.special_tokens_encoder:
            token_id = self.special_tokens_encoder[token]
        elif token in self.added_tokens_encoder:
            token_id = self.added_tokens_encoder[token]
        elif len(token) != 1:
            token_id = self.unk_token_id
        else:
            token_id = ord(token) + self._num_special_tokens
        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.special_tokens_decoder:
            token = self.special_tokens_decoder[index]
        else:
            token = chr(index - self._num_special_tokens)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        bstring = b""
        for token in tokens:
            if token in self.special_tokens_decoder:
                tok_string = self.special_tokens_decoder[token].encode("utf-8")
            elif token in self.added_tokens_decoder:
                tok_string = self.special_tokens_decoder[token].encode("utf-8")
            elif token in self.special_tokens_encoder:
                tok_string = token.encode("utf-8")
            elif token in self.added_tokens_encoder:
                tok_string = token.encode("utf-8")
            else:
                tok_string = bytes([ord(token)])
            bstring += tok_string
        string = bstring.decode("utf-8", errors="ignore")
        return string

    # ByT5Tokenizer 重写
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()

    def decode(self,
               token_ids,
               skip_special_tokens=False,
               clean_up_tokenization_spaces=True):
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.
        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.
        Args:
            token_ids (Union[List[int], Tensor]):
                List of tokenized input ids. 
            skip_special_tokens (bool, optional):
                Whether or not to remove special tokens in the decoding. Defaults to `False`.
            clean_up_tokenization_spaces (bool, optional):
                Whether or not to clean up the tokenization spaces. Defaults to `True`.
        Returns:
            str: The decoded sentence.
        """
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        text = self.convert_tokens_to_string(
            self.convert_ids_to_tokens(
                token_ids, skip_special_tokens=skip_special_tokens))
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)
        return text

    def batch_decode(self,
                     sequences,
                     skip_special_tokens=False,
                     clean_up_tokenization_spaces=True):
        """
        Convert a list of lists of token ids into a list of strings by calling decode.
        Args:
            sequences (Union[List[int], List[List[int]], Tensor]):
                List of tokenized input ids.
            skip_special_tokens (bool, optional):
                Whether or not to remove special tokens in the decoding. Defaults to `False`.
            clean_up_tokenization_spaces (bool, optional):
                Whether or not to clean up the tokenization spaces. Defaults to `True`.
        Returns:
            List[str]: The list of decoded sentences.
        """
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces)
            for seq in sequences
        ]

    @staticmethod
    def clean_up_tokenization(out_string):
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
        Args:
            out_string (str): The text to clean up.
        Returns:
            str: The cleaned-up string.
        """
        out_string = (out_string.replace(" .", ".").replace(" ?", "?")
                      .replace(" !", "!").replace(" ,", ",").replace(" ' ", "'")
                      .replace(" n't", "n't").replace(" 'm", "'m")
                      .replace(" 's", "'s").replace(" 've", "'ve")
                      .replace(" 're", "'re"))
        return out_string

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        """
        Override base class methods
        Args:
            pretrained_model_name_or_path (str): any string,only to keep same factory with other model, onlt use "ByT5tokenizer()" is ok
            *args (tuple): position arguments for model `__init__`. If provided,
                use these as position argument values for tokenizer initialization.
            **kwargs (dict): keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for tokenizer
                initialization.
        Returns:
            PretrainedTokenizer: An instance of `PretrainedTokenizer`.
        """
        tokenizer = cls()
        return tokenizer