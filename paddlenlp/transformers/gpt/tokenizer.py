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
from paddlenlp.data import Vocab
from paddlenlp.utils.log import logger

from .. import PretrainedTokenizer
from ..tokenizer_utils import convert_to_unicode, whitespace_tokenize,\
    _is_whitespace, _is_control, _is_punctuation

__all__ = [
    'GPTTokenizer',
    'GPTChineseTokenizer',
]


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


class GPTChineseTokenizer(PretrainedTokenizer):
    """
    Constructs a GPT Chinese tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.
    """
    resource_files_names = {
        "vocab_file": "vocab.json",
        "model_file": "sentencepiece.model"
    }  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "gpt-base-cn":
            "https://paddlenlp.bj.bcebos.com/models/transformers/gpt/gpt-base-cn-vocab.json",
        },
        "model_file": {
            "gpt-base-cn":
            "https://paddlenlp.bj.bcebos.com/models/transformers/gpt/gpt-base-cn-sentencepiece.model"
        }
    }
    pretrained_init_configuration = {"gpt-base-cn": {}, }

    def __init__(
            self,
            vocab_file,
            model_file,
            max_len=512,
            unk_token='<unk>',
            bod_token='<bod>',
            eod_token='<eod>',
            stop_token='\u2583',  # The token of newline.
    ):
        self._vocab_file = vocab_file
        self._model_file = model_file
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = GPTTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}
        mod = try_import("sentencepiece")
        self.sp = mod.SentencePieceProcessor(model_file=model_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")
        vocab_dict = {}
        for id in range(self.sp.get_piece_size()):
            vocab_dict[self.sp.id_to_piece(id)] = id
        self.vocab = Vocab.from_dict(vocab_dict, unk_token=unk_token)

    def tokenize(self, text):
        """
        End-to-end tokenization for GPT models.
        Args:
            text (str):
                The text to be tokenized.

        Returns:
            list[str]: A list of string representing converted tokens.
        Example:
            .. code-block::
                from paddlenlp.transformers import GPTChineseTokenizer
                tokenizer = GPTChineseTokenizer.from_pretrained('gpt-base-cn')
                print(tokenizer.tokenize('我爱祖国'))
        """
        return self._tokenize(text)

    def _tokenize(self, text):
        """ Tokenize a string. """
        seg_list = [
            x.translate(self.translator) for x in jieba.cut(text, cut_all=False)
        ]
        new_seg = " ".join(seg_list)
        return self.sp.encode(new_seg, out_type=str)

    def get_input_ids(self, text):
        tokens = self._tokenize(text)
        return self.convert_tokens_to_ids(tokens)

    @property
    def vocab_size(self):
        """
        Returns the size of vocabulary.
        Example:
            .. code-block::
                from paddlenlp.transformers import GPTChineseTokenizer
                tokenizer = GPTChineseTokenizer.from_pretrained('gpt-base-cn')
                print(tokenizer.vocab_size)
        """
        return len(self.vocab)

    def convert_ids_to_string(self, ids):
        """
        Converts a single index or a sequence of indices to a token or a
        sequence of tokens.
        Args:
            ids (int|list[int]):
                The token id (or token ids) to be converted to token(s).
        Returns:
            str|list[str]: The decoded token(s).
        """
        text = self.sp.decode(ids)
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


class GPTTokenizer(PretrainedTokenizer):
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt"
    }  # for save_pretrained
    gpt_vocab_link = "http://paddlenlp.bj.bcebos.com/models/transformers/gpt/gpt-en-vocab.json"
    gpt_merges_link = "http://paddlenlp.bj.bcebos.com/models/transformers/gpt/gpt-en-merges.txt"
    pretrained_resource_files_map = {
        "vocab_file": {
            "gpt-xlarge-en": gpt_vocab_link,
            "gpt-large-en": gpt_vocab_link,
            "gpt-medium-en": gpt_vocab_link,
            "gpt-small-en": gpt_vocab_link,
        },
        "merges_file": {
            "gpt-xlarge-en": gpt_merges_link,
            "gpt-large-en": gpt_merges_link,
            "gpt-medium-en": gpt_merges_link,
            "gpt-small-en": gpt_merges_link,
        }
    }
    pretrained_init_configuration = {
        "gpt-xlarge-en": {},
        "gpt-large-en": {},
        "gpt-medium-en": {},
        "gpt-small-en": {},
    }

    def __init__(
            self,
            vocab_file,
            merges_file,
            errors='replace',
            max_len=None,
            special_tokens=None,
            pad_token='<|endoftext|>',
            eod_token='<|endoftext|>',
            stop_token='\u010a',  # The token of newline.
    ):
        self._vocab_file = vocab_file
        self._merges_file = merges_file
        self.max_len = max_len if max_len is not None else int(1e12)
        self.num_command_tokens = 2
        self.num_type_tokens = 2

        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}

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
        return self._tokenize(text)

    def _tokenize(self, text):
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

    def get_input_ids(self, text, fn=None):
        processed_text = text
        if fn is not None:
            processed_text = fn(text)
        ids = self.convert_tokens_to_ids(self.tokenize(processed_text))
        return ids

    def convert_ids_to_string(self, ids):
        # TODO
        text = ''.join([self.decoder[id] for id in ids])
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
