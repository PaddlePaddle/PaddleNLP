# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import re
import unicodedata
from shutil import copyfile

import numpy as np
from paddle.utils import try_import

from .. import PretrainedTokenizer
from ..tokenizer_utils import convert_to_unicode, whitespace_tokenize, _is_whitespace, _is_control
from ...data.vocab import Vocab

__all__ = ['UnifiedTransformerTokenizer']


class UnifiedTransformerTokenizer(PretrainedTokenizer):
    resource_files_names = {
        "vocab_file": "vocab.txt",
        "sentencepiece_model_file": "spm.model",
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "unified_transformer-12L-cn":
            "https://paddlenlp.bj.bcebos.com/models/transformers/unified_transformer/unified_transformer-12L-cn-vocab.txt",
            "unified_transformer-12L-cn-luge":
            "https://paddlenlp.bj.bcebos.com/models/transformers/unified_transformer/unified_transformer-12L-cn-vocab.txt",
        },
        "sentencepiece_model_file": {
            "unified_transformer-12L-cn":
            "https://paddlenlp.bj.bcebos.com/models/transformers/unified_transformer/unified_transformer-12L-cn-spm.model",
            "unified_transformer-12L-cn-luge":
            "https://paddlenlp.bj.bcebos.com/models/transformers/unified_transformer/unified_transformer-12L-cn-spm.model",
        },
    }
    pretrained_init_configuration = {
        "unified_transformer-12L-cn": {
            "do_lower_case": False
        },
        "unified_transformer-12L-cn-luge": {
            "do_lower_case": False
        },
    }

    def __init__(self,
                 vocab_file,
                 sentencepiece_model_file,
                 do_lower_case=False,
                 unk_token="[UNK]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 sep_token="[SEP]",
                 mask_token="[MASK]",
                 chitchat_token="[CHAT]",
                 knowledge_token="[KNOW]",
                 recommend_token="[RECO]",
                 special_tokens_file=""):
        mod = try_import('sentencepiece')
        self.spm_model = mod.SentencePieceProcessor()

        self.do_lower_case = do_lower_case
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = ErnieTinyTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.vocab = self.load_vocabulary(
            vocab_file,
            unk_token,
            pad_token,
            cls_token,
            sep_token,
            mask_token=mask_token,
            chitchat_token=chitchat_token,
            knowledge_token=knowledge_token,
            recommend_token=recommend_token)

        # if the sentencepiece_model_file is not exists, just the default sentence-piece model 
        if os.path.isfile(sentencepiece_model_file):
            self.spm_model.Load(sentencepiece_model_file)

        pat_str = ""
        if os.path.isfile(special_tokens_file):
            self.specials = self.read_file(special_tokens_file)
            for special in self.specials:
                pat_str += "(" + re.escape(special) + ")|"
        else:
            self.specials = {}

        pat_str += r"([a-zA-Z0-9\S]+)"
        self.pat = re.compile(pat_str)

        self.vocab_file = vocab_file
        self.sentencepiece_model_file = sentencepiece_model_file

    @property
    def vocab_size(self):
        """
        return the size of vocabulary.
        Returns:
            int: the size of vocabulary.
        """
        return len(self.vocab)

    def preprocess_text(self, inputs, remove_space=True, lower=False):
        """preprocess data by removing extra space and normalize data."""
        outputs = inputs
        if remove_space:
            outputs = " ".join(inputs.strip().split())
        outputs = unicodedata.normalize("NFKD", outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if lower:
            outputs = outputs.lower()
        return outputs

    def clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        text = text.replace(u"“", u'"')\
            .replace(u'”', u'"')\
            .replace(u'‘', "'")\
            .replace(u'’', u"'")\
            .replace(u'—', u'-')
        output = []
        for char in text:
            if _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def encode_pieces(self, spm_model, text, return_unicode=True, sample=False):
        """turn sentences into word pieces."""
        # liujiaxiang: add for ernie-albert, mainly consider for “/”/‘/’/— causing too many unk
        text = self.clean_text(text)
        if not sample:
            pieces = spm_model.EncodeAsPieces(text)
        else:
            pieces = spm_model.SampleEncodeAsPieces(text, 64, 0.1)
        return pieces

    def _tokenize(self, text):
        """
        End-to-end tokenization for BERT models.
        Args:
            text (str): The text to be tokenized.
        
        Returns:
            list: A list of string representing converted tokens.
        """
        text = self.preprocess_text(text, lower=self.do_lower_case)
        tokens = []
        for match in self.pat.finditer(text):
            part_text = match.group(0)
            if part_text in self.specials:
                tokens.append(part_text)
                continue
            part_tokens = self.encode_pieces(self.spm_model, part_text)
            tokens.extend(part_tokens)
        return tokens

    def tokenize(self, text):
        """
        End-to-end tokenization for BERT models.
        Args:
            text (str): The text to be tokenized.
        
        Returns:
            list: A list of string representing converted tokens.
        """
        return self._tokenize(text)

    def merge_subword(self, tokens):
        """Merge subword."""
        ret = []
        for token in tokens:
            if token.startswith(u"▁"):
                ret.append(token[1:])
            else:
                if len(ret):
                    ret[-1] += token
                else:
                    ret.append(token)

        ret = [token for token in ret if token]
        return ret

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (list of string) in a single string. Since
        the usage of WordPiece introducing `__` to concat subwords, also remove
        `__` when converting.
        Args:
            tokens (list): A list of string representing tokens to be converted.
        Returns:
            str: Converted string from tokens.
        """
        tokens = self.merge_subword(tokens)
        out_string = " ".join(tokens).replace("<s>", "")
        out_string = out_string.replace("</s>", "\n").replace("\n ",
                                                              "\n").strip()
        return out_string

    def convert_ids_to_string(self, ids):
        """Convert ids to string."""
        tokens = self.convert_ids_to_tokens(ids)
        out_string = self.convert_tokens_to_string(tokens)
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special 
        tokens. 
        Note:
            This encodes inputs and checks the number of added tokens, and is 
            therefore not efficient. Do not put this inside your training loop.
        Args:
            pair (bool, optional): Returns the number of added tokens in the 
                case of a sequence pair if set to True, returns the number of 
                added tokens in the case of a single sequence if set to False.
                Default False.
        Returns:
            Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence by concatenating 
        and adding special tokens. 
        An UnifiedTransformer sequence has the following format:
        ::
            - single sequence: ``[CLS] X [SEP]``
            - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (list): List of IDs to which the special tokens will be 
                added.
            token_ids_1 (list, optional): Optional second list of IDs for sequence 
                pairs. Default None.
        Returns:
            list: List of input_ids with the appropriate special tokens.
        """
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        if token_ids_1 is None:
            return _cls + token_ids_0 + _sep
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        """
        Build offset map from a pair of offset map by concatenating and adding 
        offsets of special tokens.
        An UnifiedTransformer offset_mapping has the following format:
        ::
            - single sequence: ``(0,0) X (0,0)``
            - pair of sequences: `(0,0) A (0,0) B (0,0)``
        
        Args:
            offset_mapping_ids_0 (list): List of char offsets to which the special 
                tokens will be added.
            offset_mapping_ids_1 (list, optional): Optional second list of char 
                offsets for offset mapping pairs. Dafault None

        Returns:
            list: List of char offsets with the appropriate offsets of special 
                tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create the token_type_ids from the two sequences passed for the model.

        An UnifiedTransformer sequence token_type_ids has the following format:
        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If `token_ids_1` is None, this method only returns the first portion (0s).

        Args:
            token_ids_0 (list): List of IDs.
            token_ids_1 (list, optional): Optional second list of IDs for sequence 
                pairs. Default None

        Returns:
            list: List of token_type_id according to the given sequence(s).
        """
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        if token_ids_1 is None:
            return [0] * len(_cls + token_ids_0 + _sep)
        return [0] * len(_cls + token_ids_0 + _sep) + [1] * len(token_ids_1 +
                                                                _sep)

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieve sequence ids from a token list that has no special tokens added. 
        This method is called when adding special tokens using the tokenizer 
        ``prepare_for_model`` method.
        Args:
            token_ids_0 (list): List of IDs.
            token_ids_1 (list, optional): Optional second list of IDs for sequence 
                pairs. Default None.
            already_has_special_tokens (bool, optional): Whether or not the token 
                list is already formatted with special tokens for the model. Default
                False.
        Returns:
            list: A list of integers in the range [0, 1]. 1 for a special token, 
                0 for a sequence token.
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

    def save_resources(self, save_directory):
        """
        Save tokenizer related resources to files under `save_directory`.
        Args:
            save_directory (str): Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            src_path = getattr(self, name)
            save_path = os.path.join(save_directory, file_name)
            if os.path.abspath(src_path) != os.path.abspath(save_path):
                copyfile(src_path, save_path)

    @staticmethod
    def read_file(filepath):
        token_to_idx = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for num, line in enumerate(f):
                items = convert_to_unicode(line.rstrip()).split("\t")
                if len(items) > 2:
                    break
                token = items[0]
                index = int(items[1]) if len(items) == 2 else num
                token = token.strip()
                token_to_idx[token] = index
        return token_to_idx

    @staticmethod
    def load_vocabulary(filepath,
                        unk_token=None,
                        pad_token=None,
                        bos_token=None,
                        eos_token=None,
                        **kwargs):
        """
        Instantiate an instance of `Vocab` from a file reserving all tokens by 
        using `Vocab.from_dict`. The file contains a token and index of the 
        token per line, separated by '\t'.
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
        token_to_idx = UnifiedTransformerTokenizer.read_file(filepath)
        vocab = Vocab.from_dict(
            token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs)
        # Filtered the tokens that are mapped to the same id
        idx_to_token = {v: k for k, v in vocab._token_to_idx.items()}
        vocab._idx_to_token = [
            idx_to_token[idx] for idx in sorted(idx_to_token.keys())
        ]
        return vocab

    def dialogue_encode(self,
                        history,
                        response=None,
                        knowledge=None,
                        task_type='chitchat',
                        max_seq_len=512,
                        max_response_len=128,
                        max_knowledge_len=128,
                        return_position_ids=True,
                        return_token_type_ids=True,
                        return_attention_mask=True,
                        return_length=False,
                        add_start_token_as_response=False,
                        pad_to_max_seq_len=False):
        """
        Main method to encode the single-turn or multi-turn dialogue conversation. 
        It will return a dictionary containing the encoded sequence and other 
        relative informations which meets the input format requirements of the 
        UnifiedTransformer model. 
        See detail at 
        https://github.com/PaddlePaddle/Knover/tree/luge-dialogue/luge-dialogue

        Args:
            history (str|list|tuple): The history of dialogue conversation. It 
                is an utterance or list of utterances to be encoded. Each 
                utterance is a string. 
            response (str, optional): The response of dialogue conversation. 
                It should be set when training the model. It should not be set 
                when running inference. Default None.
            knowledge (str, optional): The knowledge information of dialogue 
                conversation. It should be set if the `task_type` is "knowledge" 
                or "recommend". Default None.
            task_type (str, optional): The type of dialogue conversation. It is 
                one of "chitchat", "knowledge" and "recommend". They represent 
                the chitchat dialogue, knowledge grounded dialogue and 
                conversational recommendation respectively. Default "chitchat".
            max_seq_len (int, optional): The maximum encoded sequence length.
                Default 512.
            max_response_len (int, optional): The maximum encoded sequence 
                length of the input `response`. Default 128.
            max_knowledge_len (int, optional): The maximum encoded sequence 
                length of the input `knowledge`. Default 128.
            return_position_ids (bool, optional): Whether to return the 
                position_ids. Default True.
            return_token_type_ids (bool, optional): Whether to return the 
                token_type_ids. Default True.
            return_attention_mask (bool, optional): Whether to return the 
                attention_mask. Default True.
            return_length (bool, optional): Whether to return the length of the
                encoded sequence. Default False.
            add_start_token_as_response (bool, optional): Whether to add the 
                special token [CLS] at the end of sequence as the begining of 
                the response when running inference to force the model to start 
                generating response sequence. Default False.
            pad_to_max_seq_len (bool, optional): Whether to pad the returned 
                sequences to the `max_seq_len`. Note that, in this method, 
                returned sequences will be padded on the left. Default False.
        """
        task_type_list = ["chitchat", "knowledge", "recommend"]
        # Input type checking for clearer error
        assert isinstance(history, str) or (
            isinstance(history, (list, tuple)) and
            (len(history) == 0 or len(history) != 0 and
             isinstance(history[0], str))), (
                 "The input `history` must be with type `str` (single context) "
                 "or `List[str]` (multi-turn context). But received: {}".format(
                     history))
        assert response is None or isinstance(response, str), (
            "The input `response` must of be with type `str`. But received: {}".
            format(response))
        assert knowledge is None or isinstance(knowledge, str), (
            "The input `knowledge` must of be with type `str`. But received: {}".
            format(knowledge))
        assert task_type in task_type_list, (
            "The input `task_type` must be one of {}.".format(", ".join(
                task_type_list)))
        assert max_seq_len > max_response_len + max_knowledge_len, (
            "`max_seq_len` must be greater than the sum of `max_response_len` "
            "and `max_knowledge_len`. But received `max_seq_len` is {}, "
            "`max_response_len` is {}, `max_knowledge_len` is {}.".format(
                max_seq_len, max_response_len, max_knowledge_len))
        assert response is None or not add_start_token_as_response, (
            "`add_start_token_as_response` only works when `response` is "
            "`None`. But received `add_start_token_as_response`: `{}`, "
            "`response`: {}.".format(add_start_token_as_response, response))

        knowledge_ids = []
        if knowledge is not None:
            tokens = self._tokenize(knowledge)
            knowledge_ids = self.convert_tokens_to_ids(tokens)
            if len(knowledge_ids) > max_knowledge_len - 1:
                knowledge_ids = knowledge_ids[:max_knowledge_len - 1]
            knowledge_ids += [self.sep_token_id]

        response_ids = []
        if response is not None:
            tokens = self._tokenize(response)
            response_ids = [self.cls_token_id] + self.convert_tokens_to_ids(
                tokens)
            if len(response_ids) > max_response_len - 1:
                response_ids = response_ids[:max_response_len - 1]
            response_ids += [self.sep_token_id]
        elif add_start_token_as_response:
            response_ids = [self.cls_token_id]

        special_token_id = getattr(self, task_type + '_token_id')
        knowledge_ids = [self.cls_token_id, special_token_id] + knowledge_ids
        max_history_len = max_seq_len - len(knowledge_ids) - len(response_ids)
        if isinstance(history, str):
            history = [history]
        history_ids = []
        for i in range(len(history) - 1, -1, -1):
            tokens = self._tokenize(history[i])
            if len(history_ids) + len(tokens) + 1 > max_history_len:
                if i == len(history) - 1:
                    tokens = tokens[1 - max_history_len:]
                    history_ids = (self.convert_tokens_to_ids(tokens) +
                                   [self.sep_token_id])
                break
            history_ids = (self.convert_tokens_to_ids(tokens) +
                           [self.sep_token_id]) + history_ids

        history_ids = knowledge_ids + history_ids
        # Build output dictionnary
        encoded_inputs = {}
        encoded_inputs["input_ids"] = history_ids + response_ids
        # Check lengths
        sequence_length = len(encoded_inputs["input_ids"])
        assert sequence_length <= max_seq_len

        # Considering that the logits at the last time step in the API of 
        # generative task are taken to generate the next token. In order to 
        # avoid the last time step being a pad, so take padding on the left.
        pad_length = max_seq_len - sequence_length if pad_to_max_seq_len else 0
        if pad_length > 0:
            encoded_inputs["input_ids"] = [
                self.pad_token_id
            ] * pad_length + encoded_inputs["input_ids"]

        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = [0] * len(
                history_ids) + [1] * len(response_ids)
            if pad_length > 0:
                encoded_inputs["token_type_ids"] = [
                    self.pad_token_id
                ] * pad_length + encoded_inputs["token_type_ids"]

        if return_length:
            encoded_inputs["seq_len"] = sequence_length

        if return_position_ids:
            encoded_inputs["position_ids"] = list(range(sequence_length))
            if pad_length > 0:
                encoded_inputs["position_ids"] = [
                    self.pad_token_id
                ] * pad_length + encoded_inputs["position_ids"]

        if return_attention_mask:
            attention_mask = np.ones(
                (sequence_length, sequence_length), dtype='float32') * -1e9
            start = len(history_ids)
            end = sequence_length
            attention_mask[:end, :start] = 0.0
            # Generate the lower triangular matrix using the slice of matrix
            tmp = np.triu(
                np.ones(
                    [end - start, end - start], dtype='float32') * -1e9, 1)
            attention_mask[start:end, start:end] = tmp
            encoded_inputs["attention_mask"] = attention_mask
            if pad_length > 0:
                new_mask = np.ones(
                    (max_seq_len, max_seq_len), dtype='float32') * -1e9
                new_mask[-sequence_length:, -sequence_length:] = attention_mask
                encoded_inputs["attention_mask"] = new_mask

        return encoded_inputs
