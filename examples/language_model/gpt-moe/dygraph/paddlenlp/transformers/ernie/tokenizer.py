# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.

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

import os
import pickle
import six
import shutil

from paddle.utils import try_import
from paddlenlp.utils.env import MODEL_HOME

from .. import BasicTokenizer, PretrainedTokenizer, WordpieceTokenizer

__all__ = ['ErnieTokenizer', 'ErnieTinyTokenizer']


class ErnieTokenizer(PretrainedTokenizer):
    r"""
    Constructs an ERNIE tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.

    Args:
        vocab_file (str): 
            file path of the vocabulary.
        do_lower_case (str, optional): 
            Whether the text strips accents and convert to lower case. 
            Defaults to `True`.
        unk_token (str, optional): 
            The special token for unknown words. 
            Defaults to "[UNK]".
        sep_token (str, optional): 
            The special token for separator token. 
            Defaults to "[SEP]".
        pad_token (str, optional): 
            The special token for padding. 
            Defaults to "[PAD]".
        cls_token (str, optional): 
            The special token for cls. 
            Defaults to "[CLS]".
        mask_token (str, optional): 
            The special token for mask.
            Defaults to "[MASK]".
    
    Examples:
        .. code-block:: python
            from paddlenlp.transformers import ErnieTokenizer
            tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
            encoded_inputs = tokenizer('这是一个测试样例')
            # encoded_inputs: 
            # { 
            #   'input_ids': [1, 47, 10, 7, 27, 558, 525, 314, 656, 2], 
            #   'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # }


    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-1.0":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie/vocab.txt",
            "ernie-tiny":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/vocab.txt",
            "ernie-2.0-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_v2_base/vocab.txt",
            "ernie-2.0-en-finetuned-squad":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_v2_base/vocab.txt",
            "ernie-2.0-large-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_v2_large/vocab.txt",
            "ernie-gen-base-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie-gen-base-en/vocab.txt",
            "ernie-gen-large-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie-gen-large/vocab.txt",
            "ernie-gen-large-430g-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie-gen-large-430g/vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "ernie-1.0": {
            "do_lower_case": True
        },
        "ernie-tiny": {
            "do_lower_case": True
        },
        "ernie-2.0-en": {
            "do_lower_case": True
        },
        "ernie-2.0-en-finetuned-squad": {
            "do_lower_case": True
        },
        "ernie-2.0-large-en": {
            "do_lower_case": True
        },
        "ernie-gen-base-en": {
            "do_lower_case": True
        },
        "ernie-gen-large-en": {
            "do_lower_case": True
        },
        "ernie-gen-large-430g-en": {
            "do_lower_case": True
        },
    }

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]"):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = ErnieTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=unk_token)

    @property
    def vocab_size(self):
        r"""
        return the size of vocabulary.

        Returns:
            int: the size of vocabulary.
        """
        return len(self.vocab)

    def _tokenize(self, text):
        r"""
        End-to-end tokenization for ERNIE models.

        Args:
            text (str): The text to be tokenized.
        
        Returns:
            List[str]: A list of string representing converted tokens.
        """
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def tokenize(self, text):
        r"""
        End-to-end tokenization for ERNIE models.

        Args:
            text (str): The text to be tokenized.
        
        Returns:
            List[str]: A list of string representing converted tokens.
        """
        return self._tokenize(text)

    def convert_tokens_to_string(self, tokens):
        r"""
        Converts a sequence of tokens (list of string) in a single string. Since
        the usage of WordPiece introducing `##` to concat subwords, also remove
        `##` when converting.

        Args:
            tokens (List[str]): A list of string representing tokens to be converted.

        Returns:
            str: Converted string from tokens.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        r"""
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. 
            Do not put this inside your training loop.

        Args:
            pair (str, optional): Returns the number of added tokens in the case of a sequence 
                pair if set to True, returns the number of added tokens in the case of a single sequence 
                if set to False. Defaults to False.

        Returns:
            `int`: Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        r"""
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. 
        
        A ERNIE sequence has the following format:
        ::
            - single sequence: ``[CLS] X [SEP]``
            - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.
                Defaults to `None`.

        Returns:
            List[int]: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        r"""
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens. 
        
        A ERNIE offset_mapping has the following format:
        ::
            - single sequence: ``(0,0) X (0,0)``
            - pair of sequences: `(0,0) A (0,0) B (0,0)``
        
        Args:
            offset_mapping_ids_0 (List[tuple]):
                List of char offsets to which the special tokens will be added.
            offset_mapping_ids_1 (List[tuple], optional):
                Optional second list of char offsets for offset mapping pairs.
                Defaults to `None`.

        Returns:
            List[tuple]: List of char offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        r"""
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. 

        A ERNIE sequence pair mask has the following format:
        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (List[int]):
                List of IDs.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs. 
                Defaults to `None`.

        Returns:
            List[int]: List of token_type_id according to the given sequence(s).
        """
        _sep = [self.sep_token_id]
        _cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(_cls + token_ids_0 + _sep) * [0]
        return len(_cls + token_ids_0 + _sep) * [0] + len(token_ids_1 +
                                                          _sep) * [1]


class ErnieTinyTokenizer(PretrainedTokenizer):
    r"""
    Constructs a ErnieTiny tokenizer. It uses the `dict.wordseg.pickle` cut the text to words, and 
    use the `sentencepiece` tools to cut the words to sub-words.

    Args:
        vocab_file (str): 
            The file path of the vocabulary.
        sentencepiece_model_file (str):
            The file path of sentencepice model.
        word_dict(str):
            The file path of word vocabulary, 
            which is used to do chinese word segmentation.
        do_lower_case (str, optional): 
            Whether the text strips accents and convert to lower case. 
            Defaults to `True`.
        unk_token (str, optional): 
            The special token for unknown words. 
            Defaults to "[UNK]".
        sep_token (str, optional): 
            The special token for separator token. 
            Defaults to "[SEP]".
        pad_token (str, optional): 
            The special token for padding. 
            Defaults to "[PAD]".
        cls_token (str, optional): 
            The special token for cls. 
            Defaults to "[CLS]".
        mask_token (str, optional): 
            The special token for mask.
            Defaults to "[MASK]".

    Examples:
        .. code-block:: python
            from paddlenlp.transformers import ErnieTinyTokenizer
            tokenizer = ErnieTinyTokenizer.from_pretrained('ernie-tiny')
            inputs = tokenizer('这是个测试样例')
            # inputs: 
            # {
            #   'input_ids': [3, 509, 79, 5822, 2340, 4734, 8886, 5], 
            #   'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0]
            # }
    """
    resource_files_names = {
        "sentencepiece_model_file": "spm_cased_simp_sampled.model",
        "vocab_file": "vocab.txt",
        "word_dict": "dict.wordseg.pickle"
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-tiny":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/vocab.txt"
        },
        "sentencepiece_model_file": {
            "ernie-tiny":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/spm_cased_simp_sampled.model"
        },
        "word_dict": {
            "ernie-tiny":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/dict.wordseg.pickle"
        },
    }
    pretrained_init_configuration = {"ernie-tiny": {"do_lower_case": True}}

    def __init__(self,
                 vocab_file,
                 sentencepiece_model_file,
                 word_dict,
                 do_lower_case=True,
                 encoding="utf8",
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]"):
        mod = try_import('sentencepiece')
        self.sp_model = mod.SentencePieceProcessor()
        self.word_dict = word_dict

        self.do_lower_case = do_lower_case
        self.encoding = encoding
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = ErnieTinyTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        if not os.path.isfile(word_dict):
            raise ValueError(
                "Can't find a file at path '{}'. To load the "
                "word dict from a pretrained model please use "
                "`tokenizer = ErnieTinyTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(word_dict))
        self.dict = pickle.load(open(word_dict, 'rb'))
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)

        # if the sentencepiece_model_file is not exists, just the default sentence-piece model 
        if os.path.isfile(sentencepiece_model_file):
            self.sp_model.Load(sentencepiece_model_file)

    @property
    def vocab_size(self):
        r"""
        return the size of vocabulary.

        Returns:
            int: the size of vocabulary.
        """
        return len(self.vocab)

    def cut(self, chars):
        words = []
        idx = 0
        window_size = 5
        while idx < len(chars):
            matched = False

            for i in range(window_size, 0, -1):
                cand = chars[idx:idx + i]
                if cand in self.dict:
                    words.append(cand)
                    matched = True
                    break
            if not matched:
                i = 1
                words.append(chars[idx])
            idx += i
        return words

    def _tokenize(self, text):
        r"""
        End-to-end tokenization for ErnieTiny models.

        Args:
            text (str): 
                The text to be tokenized.
        
        Returns:
            List(str): 
                A list of string representing converted tokens.
        """
        if len(text) == 0:
            return []
        if not isinstance(text, six.string_types):
            text = text.decode(self.encoding)

        text = [s for s in self.cut(text) if s != ' ']
        text = ' '.join(text)
        text = text.lower()

        tokens = self.sp_model.EncodeAsPieces(text)
        in_vocab_tokens = []
        unk_token = self.vocab.unk_token
        for token in tokens:
            if token in self.vocab:
                in_vocab_tokens.append(token)
            else:
                in_vocab_tokens.append(unk_token)
        return in_vocab_tokens

    def tokenize(self, text):
        r"""
        End-to-end tokenization for ERNIE Tiny models.

        Args:
            text (str): 
                The text to be tokenized.
        
        Returns:
            List(str): 
                A list of string representing converted tokens.
        """
        return self._tokenize(text)

    def convert_tokens_to_string(self, tokens):
        r"""
        Converts a sequence of tokens (list of string) in a single string. Since
        the usage of WordPiece introducing `##` to concat subwords, also remove
        `##` when converting.

        Args:
            tokens (list): 
                A list of string representing tokens to be converted.
        Returns:
            str: 
                Converted string from tokens.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def save_resources(self, save_directory):
        r"""
        Save tokenizer related resources to files under `save_directory`.

        Args:
            save_directory (str): Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            ### TODO: make the name 'ernie-tiny' as a variable
            source_path = os.path.join(MODEL_HOME, 'ernie-tiny', file_name)
            save_path = os.path.join(save_directory,
                                     self.resource_files_names[name])
            shutil.copyfile(source_path, save_path)

    def num_special_tokens_to_add(self, pair=False):
        r"""
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.

        Args:
            pair(str, optional): 
                Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.
                Defaults to `Fasle`.

        Returns:
            int: Number of tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        r"""
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. 
        
        A ERNIE sequence has the following format:
        ::
            - single sequence: ``[CLS] X [SEP]``
            - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.
                Defaults to `None`.

        Returns:
            List[int]: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        r"""
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens. 
        
        A ERNIE offset_mapping has the following format:
        ::
            - single sequence: ``(0,0) X (0,0)``
            - pair of sequences: `(0,0) A (0,0) B (0,0)``
        
        Args:
            offset_mapping_ids_0 (List[tuple]):
                List of char offsets to which the special tokens will be added.
            offset_mapping_ids_1 (List[tuple], optional):
                Optional second list of char offsets for offset mapping pairs.
                Defaults to `None`.

        Returns:
            List[tuple]: List of char offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        r"""
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. 

        A ERNIE sequence pair mask has the following format:
        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (List[int]):
                List of IDs.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.
                Defaults to `None`.

        Returns:
            List[int]: List of token_type_id according to the given sequence(s).
        """
        _sep = [self.sep_token_id]
        _cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(_cls + token_ids_0 + _sep) * [0]
        return len(_cls + token_ids_0 + _sep) * [0] + len(token_ids_1 +
                                                          _sep) * [1]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        r"""
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.

        Args:
            token_ids_0 (List[int]): 
                List of ids of the first sequence.
            token_ids_1 (List[int], optinal): 
                List of ids of the second sequence.
                Defaults to `None`.
            already_has_special_tokens (str, optional): 
                Whether or not the token list is already formatted with special tokens for the model. 
                Defaults to `False`.

        Returns:
            List[int]: 
                The list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
