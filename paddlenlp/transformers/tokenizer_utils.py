# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import io
import json
import os
import six
import unicodedata
from shutil import copyfile
from typing import Iterable, Iterator, Optional, List, Any, Callable, Union

from paddlenlp.utils.downloader import get_path_from_url
from paddlenlp.utils.env import MODEL_HOME

from ..data.vocab import Vocab
from .utils import InitTrackerMeta, fn_args_to_dict

__all__ = ['PretrainedTokenizer']


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming utf-8 input.
    Args:
        text (str|bytes): Text to be converted to unicode.
    Returns:
        str: converted text.
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def whitespace_tokenize(text):
    """
    Runs basic whitespace cleaning and splitting on a peice of text.
    Args:
        text (str): Text to be tokened.
    Returns:
        list(str): Token list.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_whitespace(char):
    """
    Checks whether `chars` is a whitespace character.
    """
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


@six.add_metaclass(InitTrackerMeta)
class PretrainedTokenizer(object):
    """
    The base class for all pretrained tokenizers. It mainly provides common methods
    for loading (construction and loading) and saving pretrained tokenizers. Loading
    and saving also rely on the following class attributes which should be overridden
    by derived classes accordingly:

    - **tokenizer_config_file** (str): Represents the file name of tokenizer
      configuration for configuration saving and loading in local file system.
      The value is `tokenizer_config.json`.
    - **resource_files_names** (dict): Represents resources to specific file
      names mapping for resource saving and loading in local file system. The
      keys of dict representing resource items should be argument names in
      tokenizer's `__init__` method, and the values are file names for saving
      and loading corresponding resources. The mostly used resources here are
      vocabulary file and sentence-piece model file.
    - **pretrained_init_configuration** (dict): Provides the tokenizer configurations
      of built-in pretrained tokenizers (contrasts to tokenizers in local file
      system). It has pretrained tokenizer names as keys (the same as pretrained
      model names, such as `bert-base-uncased`), and the values are dict preserving
      corresponding configuration for tokenizer initialization.
    - **pretrained_resource_files_map** (dict): Provides resource URLs of built-in
      pretrained tokenizers (contrasts to tokenizers in local file system). It
      has the same keys as `resource_files_names`, and the values are also `dict`
      mapping specific pretrained tokenizer names (such as `bert-base-uncased`)
      to corresponding resource URLs.

    Moreover, methods common to tokenizers for tokenization, token/id conversion
    and encoding as model inputs are also provided here.

    Besides, metaclass `InitTrackerMeta` is used to create `PretrainedTokenizer`,
    by which subclasses can track arguments for initialization automatically
    and expose special tokens initialization used as attributes.
    """
    tokenizer_config_file = "tokenizer_config.json"
    pretrained_init_configuration = {}
    resource_files_names = {}  # keys are arguments of __init__
    pretrained_resource_files_map = {}
    padding_side = 'right'
    pad_token_type_id = 0

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
                 max_seq_len: Optional[int]=None,
                 stride=0,
                 is_split_into_words=False,
                 pad_to_max_seq_len=False,
                 truncation_strategy="longest_first",
                 return_position_ids=False,
                 return_token_type_ids=True,
                 return_attention_mask=False,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        """
        Performs tokenization and uses the tokenized tokens to prepare model
        inputs. It supports sequence or sequence pair as input, and batch input
        is allowed. `self.encode()` or `self.batch_encode()` would be called
        separately for single or batch input depending on input format and
        `is_split_into_words` argument.

        Args:
            text (str, List[str] or List[List[str]]):
                The sequence or batch of sequences to be processed. One sequence
                is a string or a list of strings depending on whether it has been
                pretokenized. If each sequence is provided as a list of strings
                (pretokenized), you must set `is_split_into_words` as `True` to
                disambiguate with a batch of sequences.
            text_pair (str, List[str] or List[List[str]], optional):
                Same as `text` argument, while it represents for the latter
                sequence of the sequence pair.
            max_seq_len (int, optional):
                If set to a number, will limit the total sequence returned so
                that it has a maximum length. If there are overflowing tokens,
                those overflowing tokens will be added to the returned dictionary
                when `return_overflowing_tokens` is `True`. Defaults to `None`.
            stride (int, optional):
                Only available for batch input of sequence pair and mainly for
                question answering usage. When for QA, `text` represents questions
                and `text_pair` represents contexts. If `stride` is set to a
                positive number, the context will be split into multiple spans
                where `stride` defines the number of (tokenized) tokens to skip
                from the start of one span to get the next span, thus will produce
                a bigger batch than inputs to include all spans. Moreover, 'overflow_to_sample'
                and 'offset_mapping' preserving the original example and position
                information will be added to the returned dictionary. Defaults to 0.
            pad_to_max_seq_len (bool, optional):
                If set to `True`, the returned sequences would be padded up to
                `max_seq_len` specified length according to padding side
                (`self.padding_side`) and padding token id. Defaults to `False`.
            truncation_strategy (str, optional):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence
                until the input is under `max_seq_len` starting from the longest
                one at each token (when there is a pair of input sequences).
                - 'only_first': Only truncate the first sequence.
                - 'only_second': Only truncate the second sequence.
                - 'do_not_truncate': Do not truncate (raise an error if the input
                sequence is longer than `max_seq_len`).

                Defaults to 'longest_first'.
            return_position_ids (bool, optional):
                Whether to include tokens position ids in the returned dictionary.
                Defaults to `False`.
            return_token_type_ids (bool, optional):
                Whether to include token type ids in the returned dictionary.
                Defaults to `True`.
            return_attention_mask (bool, optional):
                Whether to include the attention mask in the returned dictionary.
                Defaults to `False`.
            return_length (bool, optional):
                Whether to include the length of each encoded inputs in the
                returned dictionary. Defaults to `False`.
            return_overflowing_tokens (bool, optional):
                Whether to include overflowing token information in the returned
                dictionary. Defaults to `False`.
            return_special_tokens_mask (bool, optional):
                Whether to include special tokens mask information in the returned
                dictionary. Defaults to `False`.

        Returns:
            dict or list[dict] (for batch input):
                The dict has the following optional items:

                - **input_ids** (list[int]): List of token ids to be fed to a model.
                - **position_ids** (list[int], optional): List of token position ids to be
                  fed to a model. Included when `return_position_ids` is `True`
                - **token_type_ids** (list[int], optional): List of token type ids to be
                  fed to a model. Included when `return_token_type_ids` is `True`.
                - **attention_mask** (list[int], optional): List of integers valued 0 or 1,
                  where 0 specifies paddings and should not be attended to by the
                  model. Included when `return_attention_mask` is `True`.
                - **seq_len** (int, optional): The input_ids length. Included when `return_length`
                  is `True`.
                - **overflowing_tokens** (list[int], optional): List of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **num_truncated_tokens** (int, optional): The number of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **special_tokens_mask** (list[int], optional): List of integers valued 0 or 1,
                  with 0 specifying special added tokens and 1 specifying sequence tokens.
                  Included when `return_special_tokens_mask` is `True`.
                - **offset_mapping** (list[int], optional): list of pair preserving the
                  index of start and end char in original input for each token.
                  For a special token, the index pair is `(0, 0)`. Included when
                  `stride` works.
                - **overflow_to_sample** (int, optional): Index of example from which this
                  feature is generated. Included when `stride` works.
        """
        # Input type checking for clearer error
        assert isinstance(text, str) or (
            isinstance(text, (list, tuple)) and (len(text) == 0 or (
                isinstance(text[0], str) or
                (isinstance(text[0], (list, tuple)) and
                 (len(text[0]) == 0 or isinstance(text[0][0], str)))))
        ), ("text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
            "or `List[List[str]]` (batch of pretokenized examples).")

        assert (text_pair is None or isinstance(text_pair, str) or (
            isinstance(text_pair, (list, tuple)) and (len(text_pair) == 0 or (
                isinstance(text_pair[0], str) or
                (isinstance(text_pair[0], (list, tuple)) and
                 (len(text_pair[0]) == 0 or isinstance(text_pair[0][0], str)))))
        )), (
            "text_pair input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
            "or `List[List[str]]` (batch of pretokenized examples).")

        is_batched = bool(
            (not is_split_into_words and isinstance(text, (list, tuple))) or
            (is_split_into_words and isinstance(text, (list, tuple)) and
             text and isinstance(text[0], (list, tuple))))

        if is_batched:
            batch_text_or_text_pairs = list(zip(
                text, text_pair)) if text_pair is not None else text
            return self.batch_encode(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                max_seq_len=max_seq_len,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_max_seq_len=pad_to_max_seq_len,
                truncation_strategy="longest_first",
                return_position_ids=return_position_ids,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_length=return_length,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask)
        else:
            return self.encode(
                text=text,
                text_pair=text_pair,
                max_seq_len=max_seq_len,
                pad_to_max_seq_len=pad_to_max_seq_len,
                truncation_strategy="longest_first",
                return_position_ids=return_position_ids,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_length=return_length,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask)

    @property
    def all_special_tokens(self):
        """ 
        list: All the special tokens ('<unk>', '<cls>'...) corresponding to
            special token arguments in `__init__` (arguments end with '_end').
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (
                list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self):
        """ 
        list: All the token ids corresponding to all the special tokens.
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a sequence of tokens into ids using the `vocab` attribute (an
        instance of `Vocab`). Override it if needed.

        Args：
            tokens (list[int]): List of token ids.

        Returns:
            list: Converted id list.
        """
        return self.vocab.to_indices(tokens)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (list of string) to a single string by
        using ``' '.join(tokens)`` .

        Args:
            tokens (list[str]): A sequence of tokens.

        Returns:
            str: Converted string.
        """
        return " ".join(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """
        Converts a token id or a sequence of token ids (integer) to a token or
        a sequence of tokens (str) by using the `vocab` attribute (an instance
        of `Vocab`).

        Args:
            ids (int` or `list[int]): A token id or a sequence of token ids.
            skip_special_tokens (bool, optional): Whether to skip and not
                decode special tokens when converting. Defaults to `False`.
        
        Returns:
            str: Converted token or token sequence.
        """
        tokens = self.vocab.to_tokens(ids)
        if skip_special_tokens and isinstance(tokens, list):
            tokens = [
                token for token in tokens
                if token not in self.all_special_tokens
            ]
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Creates an instance of `PretrainedTokenizer` and loads related resources
        (such as vocabulary file) according to a specific model name
        (such as `bert-base-uncased`) or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): A name of pretrained model
                for built-in tokenizers loading, such as `bert-base-uncased`.
                Or a local file directory path for local tokenizers loading.
            *args (tuple): position arguments for model `__init__`. If provided,
                use these as position argument values for tokenizer initialization.
            **kwargs (dict): keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for tokenizer
                initialization.

        Returns:
            PretrainedTokenizer: An instance of `PretrainedTokenizer`.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertTokenizer

                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        """
        pretrained_models = list(cls.pretrained_init_configuration.keys())
        vocab_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in pretrained_models:
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
            init_configuration = copy.deepcopy(
                cls.pretrained_init_configuration[
                    pretrained_model_name_or_path])
        else:
            if os.path.isdir(pretrained_model_name_or_path):
                for file_id, file_name in cls.resource_files_names.items():
                    full_file_name = os.path.join(pretrained_model_name_or_path,
                                                  file_name)
                    vocab_files[file_id] = full_file_name
                vocab_files["tokenizer_config_file"] = os.path.join(
                    pretrained_model_name_or_path, cls.tokenizer_config_file)
            else:
                raise ValueError(
                    "Calling {}.from_pretrained() with a model identifier or the "
                    "path to a directory instead. The supported model "
                    "identifiers are as follows: {}".format(
                        cls.__name__, cls.pretrained_init_configuration.keys()))

        default_root = os.path.join(MODEL_HOME, pretrained_model_name_or_path)
        resolved_vocab_files = {}
        for file_id, file_path in vocab_files.items():
            resolved_vocab_files[
                file_id] = file_path if file_path is None or os.path.isfile(
                    file_path) else get_path_from_url(file_path, default_root,
                                                      None)

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop(
            "tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with io.open(tokenizer_config_file, encoding="utf-8") as f:
                init_kwargs = json.load(f)
        else:
            init_kwargs = init_configuration
        # position args are stored in kwargs, maybe better not include
        init_args = init_kwargs.pop("init_args", ())
        init_kwargs.pop("init_class", None)

        # Update with newly provided args and kwargs
        init_args = init_args if not args else args
        init_kwargs.update(kwargs)

        # Merge resolved_vocab_files arguments in init_kwargs if not including.
        # Maybe need more ways to load resources.
        for args_name, file_path in resolved_vocab_files.items():
            # when `pretrained_model_name_or_path` is a pretrained model name,
            # use pretrained_init_configuration as `init_kwargs` to init which
            # does not include the vocab file in it, thus add vocab file into
            # args.
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path
            # when `pretrained_model_name_or_path` is a pretrained model dir,
            # use tokenizer_config_file.json as `init_kwargs` to init which
            # does include a vocab file path in it. However, if the vocab file
            # path included in json does not exist, such as was deleted, to make
            # it still work, use the vocab file under this dir.
            elif not os.path.isfile(init_kwargs[args_name]) and os.path.isfile(
                    file_path):
                init_kwargs[args_name] = file_path
        # TODO(guosheng): avoid reduplication of position args and key word args
        tokenizer = cls(*init_args, **init_kwargs)
        return tokenizer

    def save_pretrained(self, save_directory):
        """
        Save tokenizer configuration and related resources to files under
        `save_directory`. The tokenizer configuration would be saved into
        `tokenizer_config_file` indicating file (thus `tokenizer_config.json`),
        and resources would be saved into `resource_files_names` indicating files
        by using `self.save_resources(save_directory)`.
        
        The `save_directory` can be used in `from_pretrained` as argument value
        of `pretrained_model_name_or_path` to re-load the tokenizer.

        Args:
            save_directory (str): Directory to save files into.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertTokenizer

                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                tokenizer.save_pretrained('trained_model')
                # reload from save_directory
                tokenizer = BertTokenizer.from_pretrained('trained_model')
        """
        assert not os.path.isfile(
            save_directory
        ), "Saving directory ({}) should be a directory, not a file".format(
            save_directory)
        os.makedirs(save_directory, exist_ok=True)

        tokenizer_config_file = os.path.join(save_directory,
                                             self.tokenizer_config_file)
        # init_config is set in metaclass created `__init__`,
        tokenizer_config = self.init_config
        with io.open(tokenizer_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))

        self.save_resources(save_directory)

    def save_resources(self, save_directory):
        """
        Save tokenizer related resources to `resource_files_names` indicating
        files under `save_directory` by copying directly. Override it if necessary.

        Args:
            save_directory (str): Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            src_path = self.init_config[name]
            dst_path = os.path.join(save_directory, file_name)
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                copyfile(src_path, dst_path)

    @staticmethod
    def load_vocabulary(filepath,
                        unk_token=None,
                        pad_token=None,
                        bos_token=None,
                        eos_token=None,
                        **kwargs):
        """
        Instantiate an instance of `Vocab` from a file reserving all tokens
        by using `Vocab.from_dict`. The file contains a token per line, and the
        line number would be the index of corresponding token.

        Args:
            filepath (str): path of file to construct vocabulary.
            unk_token (str): special token for unknown token. If no need, it also
                could be `None`. Defaults to `None`.
            pad_token (str): special token for padding token. If no need, it also
                could be `None`. Defaults to `None`.
            bos_token (str): special token for bos token. If no need, it also
                could be `None`. Defaults to `None`.
            eos_token (str): special token for eos token. If no need, it also
                could be `None`. Defaults to `None`.
            **kwargs (dict): keyword arguments for `Vocab.from_dict`.

        Returns:
            Vocab: An instance of `Vocab`.
        """
        token_to_idx = {}
        with io.open(filepath, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                token = line.rstrip('\n')
                token_to_idx[token] = int(index)
        vocab = Vocab.from_dict(
            token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs)
        return vocab

    @staticmethod
    def save_vocabulary(filepath, vocab):
        """
        Save all tokens to a vocabulary file. The file contains a token per line,
        and the line number would be the index of corresponding token.

        Args:
            filepath (str): File path to be saved to.
            vocab (Vocab|dict): The `Vocab` or `dict` instance to be saved.
        """
        if isinstance(vocab, Vocab):
            tokens = vocab.idx_to_token
        else:
            tokens = sorted(vocab.keys(), key=lambda token: vocab[token])
        with io.open(filepath, 'w', encoding='utf-8') as f:
            for token in tokens:
                f.write(token + '\n')

    def __getattr__(self, name):
        if name.endswith('_token'):
            return self.special_tokens_map[name]
        elif name.endswith('_token_id'):
            return self.convert_tokens_to_ids(self.special_tokens_map[name[:
                                                                           -3]])
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def truncate_sequences(self,
                           ids,
                           pair_ids=None,
                           num_tokens_to_remove=0,
                           truncation_strategy='longest_first',
                           stride=0):
        """
        Truncates a sequence pair in place to the maximum length.

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to ``0``):
                number of tokens to remove using the truncation strategy
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_seq_len
                    starting from the longest one at each token (when there is a pair of input sequences).
                    Overflowing tokens only contains overflow from the first sequence.
                - 'only_first': Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_seq_len)
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_seq_len, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if truncation_strategy == 'longest_first':
            overflowing_tokens = []
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    overflowing_tokens = [ids[-1]] + overflowing_tokens
                    ids = ids[:-1]
                else:
                    pair_ids = pair_ids[:-1]
            window_len = min(len(ids), stride)
            if window_len > 0:
                overflowing_tokens = ids[-window_len:] + overflowing_tokens
        elif truncation_strategy == 'only_first':
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == 'only_second':
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
        elif truncation_strategy == 'do_not_truncate':
            raise ValueError(
                "Input sequence are too long for max_length. Please select a truncation strategy."
            )
        else:
            raise ValueError(
                "Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', 'do_not_truncate']"
            )
        return (ids, pair_ids, overflowing_tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            List[int]: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0

        return token_ids_0 + token_ids_1

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        """
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens.

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            offset_mapping_0 (List[tuple]):
                List of char offsets to which the special tokens will be added.
            offset_mapping_1 (List[tuple], optional):
                Optional second list of char offsets for offset mapping pairs.

        Returns:
            List[tuple]: List of char offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return offset_mapping_0

        return offset_mapping_0 + offset_mapping_1

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
            results (List[int]): The list of integers in the range [0, 1]:
                1 for a special token, 0 for a sequence token.
        """
        return [0] * ((len(token_ids_1)
                       if token_ids_1 else 0) + len(token_ids_0))

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.

        Should be overridden in a subclass if the model has a special way of building those.


        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (List[int]):
                List of IDs.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.

        Returns:
            List[int]: List of token_type_id according to the given sequence(s).
        """
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def num_special_tokens_to_add(self, pair):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Args:
            pair (bool, optional):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence. Defaults to `False`.
        Returns:
            int: Number of special tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def encode(self,
               text,
               text_pair=None,
               max_seq_len=512,
               pad_to_max_seq_len=False,
               truncation_strategy="longest_first",
               return_position_ids=False,
               return_token_type_ids=True,
               return_attention_mask=False,
               return_length=False,
               return_overflowing_tokens=False,
               return_special_tokens_mask=False):
        """
        Performs tokenization and uses the tokenized tokens to prepare model
        inputs. It supports sequence or sequence pair as input, and batch input
        is not allowed.

        Args:
            text (str, List[str] or List[int]):
                The sequence to be processed. One sequence is a string, a list
                of strings, or a list of integers depending on whether it has
                been pretokenized and converted to ids. 
            text_pair (str, List[str] or List[List[str]]):
                Same as `text` argument, while it represents for the latter
                sequence of the sequence pair.
            max_seq_len (int, optional):
                If set to a number, will limit the total sequence returned so
                that it has a maximum length. If there are overflowing tokens,
                those overflowing tokens will be added to the returned dictionary
                when `return_overflowing_tokens` is `True`. Defaults to `None`.
            stride (int, optional):
                Only available for batch input of sequence pair and mainly for
                question answering usage. When for QA, `text` represents questions
                and `text_pair` represents contexts. If `stride` is set to a
                positive number, the context will be split into multiple spans
                where `stride` defines the number of (tokenized) tokens to skip
                from the start of one span to get the next span, thus will produce
                a bigger batch than inputs to include all spans. Moreover, 'overflow_to_sample'
                and 'offset_mapping' preserving the original example and position
                information will be added to the returned dictionary. Defaults to 0.
            pad_to_max_seq_len (bool, optional):
                If set to `True`, the returned sequences would be padded up to
                `max_seq_len` specified length according to padding side
                (`self.padding_side`) and padding token id. Defaults to `False`.
            truncation_strategy (str, optional):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence
                until the input is under `max_seq_len` starting from the longest
                one at each token (when there is a pair of input sequences).
                - 'only_first': Only truncate the first sequence.
                - 'only_second': Only truncate the second sequence.
                - 'do_not_truncate': Do not truncate (raise an error if the input
                sequence is longer than `max_seq_len`).

                Defaults to 'longest_first'.
            return_position_ids (bool, optional):
                Whether to include tokens position ids in the returned dictionary.
                Defaults to `False`.
            return_token_type_ids (bool, optional):
                Whether to include token type ids in the returned dictionary.
                Defaults to `True`.
            return_attention_mask (bool, optional):
                Whether to include the attention mask in the returned dictionary.
                Defaults to `False`.
            return_length (bool, optional):
                Whether to include the length of each encoded inputs in the
                returned dictionary. Defaults to `False`.
            return_overflowing_tokens (bool, optional):
                Whether to include overflowing token information in the returned
                dictionary. Defaults to `False`.
            return_special_tokens_mask (bool, optional):
                Whether to include special tokens mask information in the returned
                dictionary. Defaults to `False`.

        Returns:
            dict:
                The dict has the following optional items:

                - **input_ids** (list[int]): List of token ids to be fed to a model.
                - **position_ids** (list[int], optional): List of token position ids to be
                  fed to a model. Included when `return_position_ids` is `True`
                - **token_type_ids** (list[int], optional): List of token type ids to be
                  fed to a model. Included when `return_token_type_ids` is `True`.
                - **attention_mask** (list[int], optional): List of integers valued 0 or 1,
                  where 0 specifies paddings and should not be attended to by the
                  model. Included when `return_attention_mask` is `True`.
                - **seq_len** (int, optional): The input_ids length. Included when `return_length`
                  is `True`.
                - **overflowing_tokens** (list[int], optional): List of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **num_truncated_tokens** (int, optional): The number of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **special_tokens_mask** (list[int], optional): List of integers valued 0 or 1,
                  with 0 specifying special added tokens and 1 specifying sequence tokens.
                  Included when `return_special_tokens_mask` is `True`.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self._tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        ids = get_input_ids(text)
        pair_ids = get_input_ids(text_pair) if text_pair is not None else None

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}

        # Truncation: Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(
            pair=pair))
        if max_seq_len and total_len > max_seq_len:

            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_seq_len,
                truncation_strategy=truncation_strategy, )
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_seq_len

        # Add special tokens

        sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
        token_type_ids = self.create_token_type_ids_from_sequences(ids,
                                                                   pair_ids)

        # Build output dictionnary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            encoded_inputs[
                "special_tokens_mask"] = self.get_special_tokens_mask(ids,
                                                                      pair_ids)
        if return_length:
            encoded_inputs["seq_len"] = len(encoded_inputs["input_ids"])

        # Check lengths
        assert max_seq_len is None or len(encoded_inputs[
            "input_ids"]) <= max_seq_len

        # Padding
        needs_to_be_padded = pad_to_max_seq_len and \
                             max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

        if needs_to_be_padded:
            difference = max_seq_len - len(encoded_inputs["input_ids"])
            if self.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                        "input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] +
                        [self.pad_token_type_id] * difference)
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs[
                        "special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs[
                    "input_ids"] + [self.pad_token_id] * difference
            elif self.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [
                        1
                    ] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        [self.pad_token_type_id] * difference +
                        encoded_inputs["token_type_ids"])
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [
                    self.pad_token_id
                ] * difference + encoded_inputs["input_ids"]
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                    "input_ids"])

        if return_position_ids:
            encoded_inputs["position_ids"] = list(
                range(len(encoded_inputs["input_ids"])))

        return encoded_inputs

    def batch_encode(self,
                     batch_text_or_text_pairs,
                     max_seq_len=512,
                     pad_to_max_seq_len=False,
                     stride=0,
                     is_split_into_words=False,
                     truncation_strategy="longest_first",
                     return_position_ids=False,
                     return_token_type_ids=True,
                     return_attention_mask=False,
                     return_length=False,
                     return_overflowing_tokens=False,
                     return_special_tokens_mask=False):
        """
        Performs tokenization and uses the tokenized tokens to prepare model
        inputs. It supports batch inputs of sequence or sequence pair.

        Args:
            batch_text_or_text_pairs (list):
                The element of list can be sequence or sequence pair, and the
                sequence is a string or a list of strings depending on whether
                it has been pretokenized. If each sequence is provided as a list
                of strings (pretokenized), you must set `is_split_into_words` as
                `True` to disambiguate with a sequence pair.
            max_seq_len (int, optional):
                If set to a number, will limit the total sequence returned so
                that it has a maximum length. If there are overflowing tokens,
                those overflowing tokens will be added to the returned dictionary
                when `return_overflowing_tokens` is `True`. Defaults to `None`.
            stride (int, optional):
                Only available for batch input of sequence pair and mainly for
                question answering usage. When for QA, `text` represents questions
                and `text_pair` represents contexts. If `stride` is set to a
                positive number, the context will be split into multiple spans
                where `stride` defines the number of (tokenized) tokens to skip
                from the start of one span to get the next span, thus will produce
                a bigger batch than inputs to include all spans. Moreover, 'overflow_to_sample'
                and 'offset_mapping' preserving the original example and position
                information will be added to the returned dictionary. Defaults to 0.
            pad_to_max_seq_len (bool, optional):
                If set to `True`, the returned sequences would be padded up to
                `max_seq_len` specified length according to padding side
                (`self.padding_side`) and padding token id. Defaults to `False`.
            truncation_strategy (str, optional):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence
                until the input is under `max_seq_len` starting from the longest
                one at each token (when there is a pair of input sequences).
                - 'only_first': Only truncate the first sequence.
                - 'only_second': Only truncate the second sequence.
                - 'do_not_truncate': Do not truncate (raise an error if the input
                sequence is longer than `max_seq_len`).

                Defaults to 'longest_first'.
            return_position_ids (bool, optional):
                Whether to include tokens position ids in the returned dictionary.
                Defaults to `False`.
            return_token_type_ids (bool, optional):
                Whether to include token type ids in the returned dictionary.
                Defaults to `True`.
            return_attention_mask (bool, optional):
                Whether to include the attention mask in the returned dictionary.
                Defaults to `False`.
            return_length (bool, optional):
                Whether to include the length of each encoded inputs in the
                returned dictionary. Defaults to `False`.
            return_overflowing_tokens (bool, optional):
                Whether to include overflowing token information in the returned
                dictionary. Defaults to `False`.
            return_special_tokens_mask (bool, optional):
                Whether to include special tokens mask information in the returned
                dictionary. Defaults to `False`.

        Returns:
            list[dict]:
                The dict has the following optional items:

                - **input_ids** (list[int]): List of token ids to be fed to a model.
                - **position_ids** (list[int], optional): List of token position ids to be
                  fed to a model. Included when `return_position_ids` is `True`
                - **token_type_ids** (list[int], optional): List of token type ids to be
                  fed to a model. Included when `return_token_type_ids` is `True`.
                - **attention_mask** (list[int], optional): List of integers valued 0 or 1,
                  where 0 specifies paddings and should not be attended to by the
                  model. Included when `return_attention_mask` is `True`.
                - **seq_len** (int, optional): The input_ids length. Included when `return_length`
                  is `True`.
                - **overflowing_tokens** (list[int], optional): List of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **num_truncated_tokens** (int, optional): The number of overflowing tokens.
                  Included when if `max_seq_len` is specified and `return_overflowing_tokens`
                  is True.
                - **special_tokens_mask** (list[int], optional): List of integers valued 0 or 1,
                  with 0 specifying special added tokens and 1 specifying sequence tokens.
                  Included when `return_special_tokens_mask` is `True`.
                - **offset_mapping** (list[int], optional): list of pair preserving the
                  index of start and end char in original input for each token.
                  For a sqecial token, the index pair is `(0, 0)`. Included when
                  `stride` works.
                - **overflow_to_sample** (int, optional): Index of example from which this
                  feature is generated. Included when `stride` works.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self._tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        batch_encode_inputs = []
        for example_id, tokens_or_pair_tokens in enumerate(
                batch_text_or_text_pairs):
            if not isinstance(tokens_or_pair_tokens, (list, tuple)):
                text, text_pair = tokens_or_pair_tokens, None
            elif is_split_into_words and not isinstance(
                    tokens_or_pair_tokens[0], (list, tuple)):
                text, text_pair = tokens_or_pair_tokens, None
            else:
                text, text_pair = tokens_or_pair_tokens

            first_ids = get_input_ids(text)
            second_ids = get_input_ids(
                text_pair) if text_pair is not None else None

            if stride > 0 and second_ids is not None:

                max_len_for_pair = max_seq_len - len(first_ids) - 3

                tokens = text.split()
                token_pair = text_pair.split()

                token_offset_mapping = []
                token_pair_offset_mapping = []

                token_start_offset = 0
                for token in tokens:
                    sub_tokens = []
                    for basic_token in self.basic_tokenizer.tokenize(token):
                        for sub_token in self.wordpiece_tokenizer.tokenize(
                                basic_token):
                            sub_tokens.append(sub_token if sub_token !=
                                              self.unk_token else basic_token)
                    for i in range(len(sub_tokens)):
                        if i == len(sub_tokens) - 1:
                            token_offset_mapping.append(
                                (token_start_offset, token_start_offset +
                                 len(sub_tokens[i].strip("##"))))
                            token_start_offset += (
                                len(sub_tokens[i].strip("##")) + 1)
                        else:
                            token_offset_mapping.append(
                                (token_start_offset, token_start_offset +
                                 len(sub_tokens[i].strip("##"))))
                            token_start_offset += (
                                len(sub_tokens[i].strip("##")))

                token_start_offset = 0
                for token in token_pair:
                    sub_tokens = []
                    for basic_token in self.basic_tokenizer.tokenize(token):
                        for sub_token in self.wordpiece_tokenizer.tokenize(
                                basic_token):
                            sub_tokens.append(sub_token if sub_token !=
                                              self.unk_token else basic_token)
                    for i in range(len(sub_tokens)):
                        if i == len(sub_tokens) - 1:
                            token_pair_offset_mapping.append(
                                (token_start_offset, token_start_offset +
                                 len(sub_tokens[i].strip("##"))))
                            token_start_offset += (
                                len(sub_tokens[i].strip("##")) + 1)
                        else:
                            token_pair_offset_mapping.append(
                                (token_start_offset, token_start_offset +
                                 len(sub_tokens[i].strip("##"))))
                            token_start_offset += (
                                len(sub_tokens[i].strip("##")))

                offset = 0
                while offset < len(second_ids):
                    encoded_inputs = {}
                    length = len(second_ids) - offset
                    if length > max_len_for_pair:
                        length = max_len_for_pair

                    ids = first_ids
                    pair_ids = second_ids[offset:offset + length]

                    mapping = token_offset_mapping
                    pair_mapping = token_pair_offset_mapping[offset:offset +
                                                             length]

                    offset_mapping = self.build_offset_mapping_with_special_tokens(
                        mapping, pair_mapping)
                    sequence = self.build_inputs_with_special_tokens(ids,
                                                                     pair_ids)
                    token_type_ids = self.create_token_type_ids_from_sequences(
                        ids, pair_ids)

                    # Build output dictionnary
                    encoded_inputs["input_ids"] = sequence
                    if return_token_type_ids:
                        encoded_inputs["token_type_ids"] = token_type_ids
                    if return_special_tokens_mask:
                        encoded_inputs[
                            "special_tokens_mask"] = self.get_special_tokens_mask(
                                ids, pair_ids)
                    if return_length:
                        encoded_inputs["seq_len"] = len(encoded_inputs[
                            "input_ids"])

                    # Check lengths
                    assert max_seq_len is None or len(encoded_inputs[
                        "input_ids"]) <= max_seq_len

                    # Padding
                    needs_to_be_padded = pad_to_max_seq_len and \
                                        max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

                    encoded_inputs['offset_mapping'] = offset_mapping

                    if needs_to_be_padded:
                        difference = max_seq_len - len(encoded_inputs[
                            "input_ids"])
                        if self.padding_side == 'right':
                            if return_attention_mask:
                                encoded_inputs["attention_mask"] = [1] * len(
                                    encoded_inputs[
                                        "input_ids"]) + [0] * difference
                            if return_token_type_ids:
                                # 0 for padding token mask
                                encoded_inputs["token_type_ids"] = (
                                    encoded_inputs["token_type_ids"] +
                                    [self.pad_token_type_id] * difference)
                            if return_special_tokens_mask:
                                encoded_inputs[
                                    "special_tokens_mask"] = encoded_inputs[
                                        "special_tokens_mask"] + [1
                                                                  ] * difference
                            encoded_inputs["input_ids"] = encoded_inputs[
                                "input_ids"] + [self.pad_token_id] * difference
                            encoded_inputs['offset_mapping'] = encoded_inputs[
                                'offset_mapping'] + [(0, 0)] * difference
                        elif self.padding_side == 'left':
                            if return_attention_mask:
                                encoded_inputs["attention_mask"] = [
                                    0
                                ] * difference + [1] * len(encoded_inputs[
                                    "input_ids"])
                            if return_token_type_ids:
                                # 0 for padding token mask
                                encoded_inputs["token_type_ids"] = (
                                    [self.pad_token_type_id] * difference +
                                    encoded_inputs["token_type_ids"])
                            if return_special_tokens_mask:
                                encoded_inputs["special_tokens_mask"] = [
                                    1
                                ] * difference + encoded_inputs[
                                    "special_tokens_mask"]
                            encoded_inputs["input_ids"] = [
                                self.pad_token_id
                            ] * difference + encoded_inputs["input_ids"]
                            encoded_inputs['offset_mapping'] = [
                                (0, 0)
                            ] * difference + encoded_inputs['offset_mapping']
                    else:
                        if return_attention_mask:
                            encoded_inputs["attention_mask"] = [1] * len(
                                encoded_inputs["input_ids"])

                    if return_position_ids:
                        encoded_inputs["position_ids"] = list(
                            range(len(encoded_inputs["input_ids"])))

                    encoded_inputs['overflow_to_sample'] = example_id
                    batch_encode_inputs.append(encoded_inputs)
                    if offset + length == len(second_ids):
                        break
                    offset += min(length, stride)

            else:
                batch_encode_inputs.append(
                    self.encode(
                        first_ids,
                        second_ids,
                        max_seq_len=max_seq_len,
                        pad_to_max_seq_len=pad_to_max_seq_len,
                        truncation_strategy=truncation_strategy,
                        return_position_ids=return_position_ids,
                        return_token_type_ids=return_token_type_ids,
                        return_attention_mask=return_attention_mask,
                        return_length=return_length,
                        return_overflowing_tokens=return_overflowing_tokens,
                        return_special_tokens_mask=return_special_tokens_mask))

        return batch_encode_inputs
