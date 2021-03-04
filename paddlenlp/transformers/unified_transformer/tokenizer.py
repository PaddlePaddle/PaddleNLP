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

from paddle.utils import try_import

from .. import PretrainedTokenizer
from ..tokenizer_utils import convert_to_unicode, whitespace_tokenize, _is_whitespace, _is_control
from ...data.vocab import Vocab

__all__ = ['UnifiedTransformerTokenizer']


def clean_text(text):
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


def preprocess_text(inputs, remove_space=True, lower=False):
    """preprocess data by removing extra space and normalize data."""
    outputs = inputs
    if remove_space:
        outputs = " ".join(inputs.strip().split())

    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(spm_model, text, return_unicode=True, sample=False):
    """turn sentences into word pieces."""
    # liujiaxiang: add for ernie-albert, mainly consider for “/”/‘/’/— causing too many unk
    text = clean_text(text)

    if not sample:
        pieces = spm_model.EncodeAsPieces(text)
    else:
        pieces = spm_model.SampleEncodeAsPieces(text, 64, 0.1)

    return pieces


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
                 bos_token="[CLS]",
                 eos_token="[SEP]",
                 mask_token="[MASK]",
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
            bos_token,
            eos_token,
            mask_token=mask_token)

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

    def _tokenize(self, text):
        """
        End-to-end tokenization for BERT models.
        Args:
            text (str): The text to be tokenized.
        
        Returns:
            list: A list of string representing converted tokens.
        """
        text = preprocess_text(text, lower=self.do_lower_case)
        tokens = []
        for match in self.pat.finditer(text):
            part_text = match.group(0)
            if part_text in self.specials:
                tokens.append(part_text)
                continue
            part_tokens = encode_pieces(self.spm_model, part_text)
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
        the usage of WordPiece introducing `##` to concat subwords, also remove
        `##` when converting.
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

    def convert_ids_to_str(self, ids):
        """Convert ids to string."""
        tokens = self.convert_ids_to_tokens(ids)
        out_string = self.convert_tokens_to_string(tokens)
        return out_string

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
        with open(filepath, 'r') as f:
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
