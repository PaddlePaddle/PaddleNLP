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
    The base class for all pretrained tokenizers. It provides some attributes
    and common methods for all pretrained tokenizers, including attributes for
    and special tokens (arguments of `__init__` whose name ends with `_token`)
    and methods for saving and loading.
    It also includes some class attributes (should be set by derived classes):
    - `tokenizer_config_file` (str): represents the file name for saving and loading
      tokenizer configuration, it's value is `tokenizer_config.json`.
    - `resource_files_names` (dict): use this to map resource related arguments
      of `__init__` to specific file names for saving and loading.
    - `pretrained_resource_files_map` (dict): The dict has the same keys as
      `resource_files_names`, the values are also dict mapping specific pretrained
      model name to URL linking to vocabulary or other resources.
    - `pretrained_init_configuration` (dict): The dict has pretrained model names
      as keys, and the values are also dict preserving corresponding configuration
      for tokenizer initialization.
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
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences. This method will call `self.encode()` or `self.batch_encode()` depending on input format and  
        `is_split_into_words` argument.

        Args:
            text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
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
        List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
        (cls_token, unk_token...).
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
        List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
        class attributes (cls_token, unk_token...).
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a sequence of tokens into ids using the vocab. The tokenizer
        should has the `vocab` attribute.
        Args：
            tokens (list(str)): List of tokens.
        Returns:
            list: Converted id list.
        """
        return self.vocab.to_indices(tokens)

    def convert_tokens_to_string(self, tokens):
        """ 
        Converts a sequence of tokens (list of string) to a single string by
        using :code:`' '.join(tokens)` .
        Args:
            tokens (list(str)): List of tokens.
        Returns:
            str: Converted string.
        """
        return " ".join(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """
        Converts a single index or a sequence of indices (integers) in a token
        or a sequence of tokens (str) by using the vocabulary.

        Args:
            skip_special_tokens: Don't decode special tokens (self.all_special_tokens).
                Default: False
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
        Instantiate an instance of `PretrainedTokenizer` from a predefined
        tokenizer specified by name or path., and it always corresponds to a
        pretrained model.
        Args:
            pretrained_model_name_or_path (str): A name of or a file path to a
                pretrained model.
            *args (tuple): position arguments for `__init__`. If provide, use
                this as position argument values for tokenizer initialization.
            **kwargs (dict): keyword arguments for `__init__`. If provide, use
                this to update pre-defined keyword argument values for tokenizer
                initialization.
        Returns:
            PretrainedTokenizer: An instance of PretrainedTokenizer.
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
        `save_directory`.
        Args:
            save_directory (str): Directory to save files into.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving directory ({}) should be a directory".format(save_directory)
        tokenizer_config_file = os.path.join(save_directory,
                                             self.tokenizer_config_file)
        # init_config is set in metaclass created `__init__`,
        tokenizer_config = self.init_config
        with io.open(tokenizer_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokenizer_config, ensure_ascii=False))

        self.save_resources(save_directory)

    def save_resources(self, save_directory):
        """
        Save tokenizer related resources to files under `save_directory`.
        Args:
            save_directory (str): Directory to save files into.
        """
        assert hasattr(self, 'vocab') and len(
            self.resource_files_names) == 1, "Must overwrite `save_resources`"
        file_name = os.path.join(save_directory,
                                 list(self.resource_files_names.values())[0])
        self.save_vocabulary(file_name, self.vocab)

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
                could be None. Default: None.
            pad_token (str): special token for padding token. If no need, it also
                could be None. Default: None.
            bos_token (str): special token for bos token. If no need, it also
                could be None. Default: None.
            eos_token (str): special token for eos token. If no need, it also
                could be None. Default: None.
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
        Agrs:
            filepath (str): File path to be saved to.
            vocab (Vocab|dict): the Vocab or dict instance to be saved.
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
            :obj:`List[int]`: List of input_id with the appropriate special tokens.
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
            offset_mapping_ids_0 (:obj:`List[tuple]`):
                List of char offsets to which the special tokens will be added.
            offset_mapping_ids_1 (:obj:`List[tuple]`, `optional`):
                Optional second list of char offsets for offset mapping pairs.

        Returns:
            :obj:`List[tuple]`: List of char offsets with the appropriate offsets of special tokens.
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
            token_ids_1 (List[int], optinal): List of ids of the second sequence.
            already_has_special_tokens (bool, optional): Whether or not the token list is already 
                formatted with special tokens for the model. Defaults to None.

        Returns:
            results (List[int]): The list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        return [0] * ((len(token_ids_1)
                       if token_ids_1 else 0) + len(token_ids_0))

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. 

        Should be overridden in a subclass if the model has a special way of building those.
        

        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of token_type_id according to the given sequence(s).
        """
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

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
        Returns a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_seq_len`` is specified.

        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            max_seq_len (:obj:`int`, `optional`, defaults to :int:`512`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            pad_to_max_seq_len (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the
                model's max length.
            truncation_strategy (:obj:`str`, `optional`, defaults to `longest_first`):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_seq_len
                  starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_seq_len)
            return_position_ids (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return tokens position ids (default True).
            return_token_type_ids (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to return token type IDs.
            return_attention_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to return the attention mask.
            return_length (:obj:`int`, defaults to :obj:`False`):
                If set the resulting dictionary will include the length of each encoded inputs
            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return overflowing token information (default False).
            return_special_tokens_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return special tokens mask information (default False).

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    position_ids: list[int] if return_position_ids is True 
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    attention_mask: list[int] if return_attention_mask is True 
                    seq_len: int if return_length is True 
                    overflowing_tokens: list[int] if a ``max_seq_len`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_seq_len`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if return_special_tokens_mask is True
                }

            With the fields:

            - ``input_ids``: list of token ids to be fed to a model
            - ``position_ids``: list of token position ids to be fed to a model
            - ``token_type_ids``: list of token type ids to be fed to a model
            - ``attention_mask``: list of indices specifying which tokens should be attended to by the model
            - ``length``: the input_ids length
            - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
            - ``num_truncated_tokens``: number of overflowing tokens a ``max_seq_len`` is specified
            - ``special_tokens_mask``: list of [0, 1], with 0 specifying special added
              tokens and 1 specifying sequence tokens.
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
        Returns a list of dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_seq_len`` is specified.

        Args:
            batch_text_or_text_pairs (:obj:`List[str]`, :obj:`List[Tuple[str, str]]`, :obj:`List[List[str]]`, :obj:`List[Tuple[List[str], List[str]]]`, :obj:`List[List[int]]`, :obj:`List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of
                string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence
            max_seq_len (:obj:`int`, `optional`, defaults to :int:`512`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            pad_to_max_seq_len (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the
                model's max length.
            stride (:obj:`int`, `optional`, defaults to 0):
                If set to a positive number and batch_text_or_text_pairs is a list of pair sequences, the overflowing 
                tokens which contain some tokens from the end of the truncated second sequence will be concatenated with 
                the first sequence to generate new features. And The overflowing tokens would not be returned in dictionary.
                The value of this argument defines the number of overlapping tokens.
            is_split_into_words (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the text has been pretokenized.
            truncation_strategy (:obj:`str`, `optional`, defaults to `longest_first`):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_seq_len
                  starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_seq_len)
            return_position_ids (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return tokens position ids (default True).
            return_token_type_ids (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to return token type IDs.
            return_attention_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to return the attention mask.
            return_length (:obj:`int`, defaults to :obj:`False`):
                If set the resulting dictionary will include the length of each encoded inputs
            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return overflowing token information (default False).
            return_special_tokens_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return special tokens mask information (default False).

        Return:
            A List of dictionary of shape::

                {
                    input_ids: list[int],
                    position_ids: list[int] if return_position_ids is True 
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    attention_mask: list[int] if return_attention_mask is True 
                    seq_len: int if return_length is True 
                    overflowing_tokens: list[int] if a ``max_seq_len`` is specified and return_overflowing_tokens is True and stride is 0
                    num_truncated_tokens: int if a ``max_seq_len`` is specified and return_overflowing_tokens is True and stride is 0
                    special_tokens_mask: list[int] if return_special_tokens_mask is True
                    offset_mapping: list[Tuple] if stride is a positive number and batch_text_or_text_pairs is a list of pair sequences
                    overflow_to_sample: int if stride is a positive number and batch_text_or_text_pairs is a list of pair sequences
                }

            With the fields:

            - ``input_ids``: list of token ids to be fed to a model
            - ``position_ids``: list of token position ids to be fed to a model
            - ``token_type_ids``: list of token type ids to be fed to a model
            - ``attention_mask``: list of indices specifying which tokens should be attended to by the model
            - ``length``: the input_ids length
            - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
            - ``num_truncated_tokens``: number of overflowing tokens a ``max_seq_len`` is specified
            - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
              tokens and 1 specifying sequence tokens.
            - ``offset_mapping``: list of (index of start char in text,index of end char in text) of token. (0,0) if token is a sqecial token
            - ``overflow_to_sample``: index of example from which this feature is generated
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
