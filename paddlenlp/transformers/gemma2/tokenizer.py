#
#
#
"""Tokenization classes for Gemma2."""

import os
import re
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece as spm

from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.utils.log import logger

__all__ = ["Gemma2Tokenizer"]

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/gemma-2-2b": "https://bj.bcebos.com/paddlenlp/models/transformers/gemma2/gemma-2-2b/tokenizer.model",
        "google/gemma-2-9b": "https://bj.bcebos.com/paddlenlp/models/transformers/gemma2/gemma-2-9b/tokenizer.model",
        "google/gemma-2-27b": "https://bj.bcebos.com/paddlenlp/models/transformers/gemma2/gemma-2-27b/tokenizer.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/gemma-2-2b": 8192,
    "google/gemma-2-9b": 8192,
    "google/gemma-2-27b": 8192,
}

SPIECE_UNDERLINE = "▁"


class Gemma2Tokenizer(PretrainedTokenizer):
    """
    Construct a Gemma2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<bos>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<eos>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding.
        sp_model_kwargs (`Dict[str, Any]`, *optional*):
            Arguments to be passed to the `SentencePieceProcessor` object. Can be used to set special tokens. It can be
            used to pass special tokens like `bos_id`, `eos_id`, `pad_id`, `unk_id`, `mask_id`.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether to add the bos_token at the beginning of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether to add the eos_token at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to clean up the tokenization spaces.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not to use the default system prompt.
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to add spaces between special tokens.
        legacy (`bool`, *optional*, defaults to `False`):
            Whether or not to use the legacy tokenization style.
        chat_template (`str`, *optional*):
            The template to use for chat completion.
    """

    resource_files_names = VOCAB_FILES_NAMES
    pretrained_resource_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = {}
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

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
        legacy=False,
        chat_template=None,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.legacy = legacy
        self.chat_template = chat_template

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text, **kwargs):
        """Returns a tokenized string."""
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        if add_prefix_space and not text.startswith(" "):
            text = " " + text

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
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

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

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Gemma2 sequence has the following format:

        - single sequence: `[BOS] X [EOS]` (with `add_bos_token=True, add_eos_token=True`)
        - single sequence: `X [EOS]` (with `add_bos_token=False, add_eos_token=True`)
        - single sequence: `[BOS] X` (with `add_bos_token=True, add_eos_token=False`)
        - single sequence: `X` (with `add_bos_token=False, add_eos_token=False`)
        - pair of sequences: `[BOS] A [EOS] B [EOS]` (with `add_bos_token=True, add_eos_token=True`)
        - pair of sequences: `A [EOS] B [EOS]` (with `add_bos_token=False, add_eos_token=True`)
        - pair of sequences: `[BOS] A B` (with `add_bos_token=True, add_eos_token=False`)
        - pair of sequences: `A B` (with `add_bos_token=False, add_eos_token=False`)

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            if self.add_bos_token and self.add_eos_token:
                return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
            elif self.add_bos_token:
                return [self.bos_token_id] + token_ids_0
            elif self.add_eos_token:
                return token_ids_0 + [self.eos_token_id]
            else:
                return token_ids_0
        else:
            if self.add_bos_token and self.add_eos_token:
                return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
            elif self.add_bos_token:
                return [self.bos_token_id] + token_ids_0 + token_ids_1
            elif self.add_eos_token:
                return token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
            else:
                return token_ids_0 + token_ids_1

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
            if self.add_bos_token and self.add_eos_token:
                return [1] + ([0] * len(token_ids_0)) + [1]
            elif self.add_bos_token:
                return [1] + ([0] * len(token_ids_0))
            elif self.add_eos_token:
                return ([0] * len(token_ids_0)) + [1]
            else:
                return [0] * len(token_ids_0)
        else:
            if self.add_bos_token and self.add_eos_token:
                return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
            elif self.add_bos_token:
                return [1] + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1))
            elif self.add_eos_token:
                return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
            else:
                return [0] * (len(token_ids_0) + len(token_ids_1))

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. Gemma2 does not
        make use of token type ids, therefore a list of zeros is returned.

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
            if self.add_bos_token and self.add_eos_token:
                return [0] * (len(token_ids_0) + 2)
            elif self.add_bos_token or self.add_eos_token:
                return [0] * (len(token_ids_0) + 1)
            else:
                return [0] * len(token_ids_0)
        else:
            if self.add_bos_token and self.add_eos_token:
                return [0] * (len(token_ids_0) + len(token_ids_1) + 3)
            elif self.add_bos_token or self.add_eos_token:
                return [0] * (len(token_ids_0) + len(token_ids_1) + 2)
            else:
                return [0] * (len(token_ids_0) + len(token_ids_1) + 1)

    def encode_chat_inputs(
        self,
        conversations: List[List[str]],
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Tuple[List[int], List[int]]]]:
        """
        Encode a list of conversations into a list of token IDs for training.

        Args:
            conversations (`List[List[str]]`):
                List of conversations, where each conversation is a list of strings.
            context_data (`Dict[str, Any]`, *optional*):
                Additional context data for the conversations.

        Returns:
            `Dict[str, List[Tuple[List[int], List[int]]]]`: Dictionary containing the encoded conversations.
        """
        if self.chat_template is None:
            raise ValueError("Chat template is not set for this tokenizer.")

        is_training = context_data.get("is_training", False) if context_data is not None else False

        encoded_conversations = []
        for conversation in conversations:
            encoded_rounds = []
            for i in range(0, len(conversation), 2):
                if i + 1 < len(conversation):
                    user_message = conversation[i]
                    assistant_message = conversation[i + 1]

                    user_tokens = self.encode(
                        f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n",
                        add_special_tokens=False,
                    )["input_ids"]
                    assistant_tokens = self.encode(
                        f"{assistant_message}<end_of_turn>\n", add_special_tokens=False
                    )["input_ids"]

                    if i == 0 and self.add_bos_token:
                        user_tokens = [self.bos_token_id] + user_tokens

                    encoded_rounds.append((user_tokens, assistant_tokens))

            encoded_conversations.append(encoded_rounds)

        return {"conversations": [item for sublist in encoded_conversations for item in sublist]}
