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

"""Tokenization class for SpeechT5."""


import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from paddlenlp.transformers import PretrainedTokenizer

from ...utils.log import logger

VOCAB_FILES_NAMES = {"vocab_file": "spm_char.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/speecht5_asr": "https://huggingface.co/microsoft/speecht5_asr/resolve/main/spm_char.model",
        "microsoft/speecht5_tts": "https://huggingface.co/microsoft/speecht5_tts/resolve/main/spm_char.model",
        "microsoft/speecht5_vc": "https://huggingface.co/microsoft/speecht5_vc/resolve/main/spm_char.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/speecht5_asr": 1024,
    "microsoft/speecht5_tts": 1024,
    "microsoft/speecht5_vc": 1024,
}


__all__ = ["SpeechT5Tokenizer"]


# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]


class SpeechT5Tokenizer(PretrainedTokenizer):
    """
    Construct a SpeechT5 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The begin of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    resource_files_names = VOCAB_FILES_NAMES
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_resource_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        self._in_target_context_manager = False

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
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
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        suffix_ones = [1]
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + suffix_ones
        return ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
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

    # def __call__(
    #     self,
    #     text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
    #     text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
    #     text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
    #     text_pair_target: Optional[
    #         Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
    #     ] = None,
    #     add_special_tokens: bool = True,
    #     padding: Union[bool, str, PaddingStrategy] = False,
    #     truncation: Union[bool, str, TruncationStrategy] = None,
    #     max_length: Optional[int] = None,
    #     stride: int = 0,
    #     is_split_into_words: bool = False,
    #     pad_to_multiple_of: Optional[int] = None,
    #     return_tensors: Optional[str] = None,
    #     return_token_type_ids: Optional[bool] = None,
    #     return_attention_mask: Optional[bool] = None,
    #     return_overflowing_tokens: bool = False,
    #     return_special_tokens_mask: bool = False,
    #     return_offsets_mapping: bool = False,
    #     return_length: bool = False,
    #     verbose: bool = True,
    #     **kwargs,
    # ) -> BatchEncoding:
    #     """
    #     Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
    #     sequences.

    #     Args:
    #         text (`str`, `List[str]`, `List[List[str]]`, *optional*):
    #             The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
    #             (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
    #             `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
    #         text_pair (`str`, `List[str]`, `List[List[str]]`, *optional*):
    #             The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
    #             (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
    #             `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
    #         text_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
    #             The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
    #             list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
    #             you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
    #         text_pair_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
    #             The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
    #             list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
    #             you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
    #     """
    #     # To avoid duplicating
    #     all_kwargs = {
    #         "add_special_tokens": add_special_tokens,
    #         "padding": padding,
    #         "truncation": truncation,
    #         "max_length": max_length,
    #         "stride": stride,
    #         "is_split_into_words": is_split_into_words,
    #         "pad_to_multiple_of": pad_to_multiple_of,
    #         "return_tensors": return_tensors,
    #         "return_token_type_ids": return_token_type_ids,
    #         "return_attention_mask": return_attention_mask,
    #         "return_overflowing_tokens": return_overflowing_tokens,
    #         "return_special_tokens_mask": return_special_tokens_mask,
    #         "return_offsets_mapping": return_offsets_mapping,
    #         "return_length": return_length,
    #         "verbose": verbose,
    #     }
    #     all_kwargs.update(kwargs)
    #     if text is None and text_target is None:
    #         raise ValueError("You need to specify either `text` or `text_target`.")
    #     if text is not None:
    #         # The context manager will send the inputs as normal texts and not text_target, but we shouldn't change the
    #         # input mode in this case.
    #         if not self._in_target_context_manager:
    #             self._switch_to_input_mode()
    #         encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
    #     if text_target is not None:
    #         self._switch_to_target_mode()
    #         target_encodings = self._call_one(text=text_target, text_pair=text_pair_target, **all_kwargs)
    #     # Leave back tokenizer in input mode
    #     self._switch_to_input_mode()

    #     if text_target is None:
    #         return encodings
    #     elif text is None:
    #         return target_encodings
    #     else:
    #         encodings["labels"] = target_encodings["input_ids"]
    #         return encodings

    # def _call_one(
    #     self,
    #     text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
    #     text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
    #     add_special_tokens: bool = True,
    #     padding: Union[bool, str, PaddingStrategy] = False,
    #     truncation: Union[bool, str, TruncationStrategy] = None,
    #     max_length: Optional[int] = None,
    #     stride: int = 0,
    #     is_split_into_words: bool = False,
    #     pad_to_multiple_of: Optional[int] = None,
    #     return_tensors: Optional[str] = None,
    #     return_token_type_ids: Optional[bool] = None,
    #     return_attention_mask: Optional[bool] = None,
    #     return_overflowing_tokens: bool = False,
    #     return_special_tokens_mask: bool = False,
    #     return_offsets_mapping: bool = False,
    #     return_length: bool = False,
    #     verbose: bool = True,
    #     **kwargs,
    # ) -> BatchEncoding:
    #     # Input type checking for clearer error
    #     def _is_valid_text_input(t):
    #         if isinstance(t, str):
    #             # Strings are fine
    #             return True
    #         elif isinstance(t, (list, tuple)):
    #             # List are fine as long as they are...
    #             if len(t) == 0:
    #                 # ... empty
    #                 return True
    #             elif isinstance(t[0], str):
    #                 # ... list of strings
    #                 return True
    #             elif isinstance(t[0], (list, tuple)):
    #                 # ... list with an empty list or with a list of strings
    #                 return len(t[0]) == 0 or isinstance(t[0][0], str)
    #             else:
    #                 return False
    #         else:
    #             return False

    #     if not _is_valid_text_input(text):
    #         raise ValueError(
    #             "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
    #             "or `List[List[str]]` (batch of pretokenized examples)."
    #         )

    #     if text_pair is not None and not _is_valid_text_input(text_pair):
    #         raise ValueError(
    #             "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
    #             "or `List[List[str]]` (batch of pretokenized examples)."
    #         )

    #     if is_split_into_words:
    #         is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
    #     else:
    #         is_batched = isinstance(text, (list, tuple))

    #     if is_batched:
    #         if isinstance(text_pair, str):
    #             raise TypeError(
    #                 "when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as"
    #                 " `text`."
    #             )
    #         if text_pair is not None and len(text) != len(text_pair):
    #             raise ValueError(
    #                 f"batch length of `text`: {len(text)} does not match batch length of `text_pair`:"
    #                 f" {len(text_pair)}."
    #             )
    #         batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
    #         return self.batch_encode_plus(
    #             batch_text_or_text_pairs=batch_text_or_text_pairs,
    #             add_special_tokens=add_special_tokens,
    #             padding=padding,
    #             truncation=truncation,
    #             max_length=max_length,
    #             stride=stride,
    #             is_split_into_words=is_split_into_words,
    #             pad_to_multiple_of=pad_to_multiple_of,
    #             return_tensors=return_tensors,
    #             return_token_type_ids=return_token_type_ids,
    #             return_attention_mask=return_attention_mask,
    #             return_overflowing_tokens=return_overflowing_tokens,
    #             return_special_tokens_mask=return_special_tokens_mask,
    #             return_offsets_mapping=return_offsets_mapping,
    #             return_length=return_length,
    #             verbose=verbose,
    #             **kwargs,
    #         )
    #     else:
    #         return self.encode_plus(
    #             text=text,
    #             text_pair=text_pair,
    #             add_special_tokens=add_special_tokens,
    #             padding=padding,
    #             truncation=truncation,
    #             max_length=max_length,
    #             stride=stride,
    #             is_split_into_words=is_split_into_words,
    #             pad_to_multiple_of=pad_to_multiple_of,
    #             return_tensors=return_tensors,
    #             return_token_type_ids=return_token_type_ids,
    #             return_attention_mask=return_attention_mask,
    #             return_overflowing_tokens=return_overflowing_tokens,
    #             return_special_tokens_mask=return_special_tokens_mask,
    #             return_offsets_mapping=return_offsets_mapping,
    #             return_length=return_length,
    #             verbose=verbose,
    #             **kwargs,
    #         )

    # def encode(
    #     self,
    #     text: Union[TextInput, PreTokenizedInput, EncodedInput],
    #     text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
    #     add_special_tokens: bool = True,
    #     padding: Union[bool, str, PaddingStrategy] = False,
    #     truncation: Union[bool, str, TruncationStrategy] = None,
    #     max_length: Optional[int] = None,
    #     stride: int = 0,
    #     return_tensors: Optional[str] = None,
    #     **kwargs,
    # ) -> List[int]:
    #     """
    #     Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

    #     Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

    #     Args:
    #         text (`str`, `List[str]` or `List[int]`):
    #             The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
    #             `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
    #             method).
    #         text_pair (`str`, `List[str]` or `List[int]`, *optional*):
    #             Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
    #             the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
    #             method).
    #     """
    #     encoded_inputs = self.encode_plus(
    #         text,
    #         text_pair=text_pair,
    #         add_special_tokens=add_special_tokens,
    #         padding=padding,
    #         truncation=truncation,
    #         max_length=max_length,
    #         stride=stride,
    #         return_tensors=return_tensors,
    #         **kwargs,
    #     )

    #     return encoded_inputs["input_ids"]
