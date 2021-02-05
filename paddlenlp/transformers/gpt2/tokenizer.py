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

import os
from functools import lru_cache
from collections import namedtuple

import json
import jieba
import shutil
from paddle.utils import try_import

from .. import PretrainedTokenizer
from ..tokenizer_utils import convert_to_unicode, whitespace_tokenize,\
    _is_whitespace, _is_control, _is_punctuation

__all__ = [
    'GPT2Tokenizer',
    'GPT2ChineseTokenizer',
]

COMMAND_TUPLE = namedtuple('CommandToken', ('name', 'token', 'Id'))
TYPE_TUPLE = namedtuple('TypeToken', ('name', 'token', 'Id'))


class CommandToken(object):
    def __init__(self, name, token, Id):
        self.name = name
        self.token = token
        self.Id = Id

    def __str__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    _chr = chr
    bs = list(range(ord("!"), ord("~") + 1)) + list(
        range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class GPT2ChineseTokenizer(PretrainedTokenizer):
    """
    Constructs a GPT2 Chinese tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.
    """
    resource_files_names = {
        "vocab_file": "vocab.json",
        "model_file": "sentencepiece.model"
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "gpt2-base-cn":
            "https://paddlenlp.bj.bcebos.com/models/transformers/gpt2/gpt2-base-cn-vocab.json",
        },
        "model_file": {
            "gpt2-base-cn":
            "https://paddlenlp.bj.bcebos.com/models/transformers/gpt2/gpt2-base-cn-sentencepiece.model"
        }
    }
    pretrained_init_configuration = {"gpt2-base-cn": {"do_lower_case": True}, }

    def __init__(self,
                 vocab_file,
                 model_file,
                 do_lower_case=True,
                 max_len=512,
                 bod_id="<bod>",
                 eod_id="<eod>",
                 max_length=None):
        self._vocab_file = vocab_file
        self._model_file = model_file
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}
        mod = try_import("sentencepiece")
        self.sp = mod.SentencePieceProcessor(model_file=model_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")

    def tokenize(self, text):
        """ Tokenize a string. """
        seg_list = [
            x.translate(self.translator) for x in jieba.cut(text, cut_all=False)
        ]
        new_seg = " ".join(seg_list)
        return self.sp.encode(new_seg)

    def encode(self, text):
        return self.convert_tokens_to_ids(text)

    def decode(self, tokens):
        return self.convert_ids_to_tokens(tokens)

    def convert_tokens_to_ids(self, text):
        res = self.tokenize(text)
        return res

    def convert_ids_to_tokens(self, tokens):
        text = self.sp.decode(tokens)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583',
                                                                    '\n')
        return text

    def save_resources(self, save_directory):
        """
        Save tokenizer related resources to files under `save_directory`.
        Args:
            save_directory (str): Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            save_path = os.path.join(save_directory, file_name)
            shutil.copyfile(getattr(self, "_%s" % name), save_path)


class GPT2Tokenizer(PretrainedTokenizer):
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt"
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "gpt2-large-en":
            "http://paddlenlp.bj.bcebos.com/models/transformers/gpt2/gpt2-large-en-vocab.json",
            "gpt2-medium-en":
            "http://paddlenlp.bj.bcebos.com/models/transformers/gpt2/gpt2-medium-en-vocab.json",
            "gpt2-small-en":
            "http://paddlenlp.bj.bcebos.com/models/transformers/gpt2/gpt2-small-en-vocab.json",
        },
        "merges_file": {
            "gpt2-large-en":
            "http://paddlenlp.bj.bcebos.com/models/transformers/gpt2/gpt2-large-en-merges.txt",
            "gpt2-medium-en":
            "http://paddlenlp.bj.bcebos.com/models/transformers/gpt2/gpt2-medium-en-merges.txt",
            "gpt2-small-en":
            "http://paddlenlp.bj.bcebos.com/models/transformers/gpt2/gpt2-small-en-merges.txt",
        }
    }
    pretrained_init_configuration = {
        "gpt2-large-en": {
            "do_lower_case": True
        },
        "gpt2-medium-en": {
            "do_lower_case": True
        },
        "gpt2-small-en": {
            "do_lower_case": True
        },
    }

    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors='replace',
                 special_tokens=None,
                 max_len=None,
                 do_lower_case=True):
        self._vocab_file = vocab_file
        self._merges_file = merges_file
        self.max_len = int(1e12)
        self.num_command_tokens = 2
        self.num_type_tokens = 2

        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}

        # construct the command tokens
        self._command_tokens = [
            CommandToken('pad', '<|endoftext|>', self.encoder['<|endoftext|>']),
            CommandToken('eod', '<|endoftext|>', self.encoder['<|endoftext|>']),
        ]
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {
            tok.token: tok
            for tok in self._command_tokens
        }
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}

        self.num_tokens = len(self.encoder)
        self.num_text_tokens = self.num_tokens - 1
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        re = try_import("regex")
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.set_special_tokens(special_tokens)

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    def set_special_tokens(self, special_tokens):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, len(self.encoder) + i)
                                   for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {
            v: k
            for k, v in self.special_tokens.items()
        }
        logger.info("Special tokens {}".format(self.special_tokens))

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i +
                                                                   1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        re = try_import("regex")
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(
                bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        if isinstance(tokens, str):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, 0)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, 0))
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".
                format(len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        tokens = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                tokens.append(self.decoder[i])
        return tokens

    def encode(self, text, fn=None):
        processed_text = text
        if fn is not None:
            processed_text = fn(text)
        ids = self.convert_tokens_to_ids(self.tokenize(processed_text))
        return ids

    def decode(self, tokens):
        # TODO
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors=self.errors)
        return text

    def save_resources(self, save_directory):
        """
        Save tokenizer related resources to files under `save_directory`.
        Args:
            save_directory (str): Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            save_path = os.path.join(save_directory, file_name)
            shutil.copyfile(getattr(self, "_%s" % name), save_path)
