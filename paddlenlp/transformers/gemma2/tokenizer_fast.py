#
#
#
"""Fast Tokenization classes for Gemma2."""

import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

from paddlenlp.transformers.tokenizer_utils_fast import PretrainedFastTokenizer

from .tokenizer import PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES, PRETRAINED_VOCAB_FILES_MAP, VOCAB_FILES_NAMES, Gemma2Tokenizer


class Gemma2TokenizerFast(PretrainedFastTokenizer):
    """
    Construct a "fast" Gemma2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<bos>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<eos>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding.
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
    slow_tokenizer_class = Gemma2Tokenizer
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=False,
        spaces_between_special_tokens=False,
        legacy=False,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.legacy = legacy
        self.chat_template = chat_template

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
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if self.slow_tokenizer.vocab_file is not None:
            copyfile(self.slow_tokenizer.vocab_file, vocab_file)
        elif self.slow_tokenizer.sp_model is not None:
            with open(vocab_file, "wb") as f:
                f.write(self.slow_tokenizer.sp_model.serialized_model_proto())

        return (vocab_file,)
