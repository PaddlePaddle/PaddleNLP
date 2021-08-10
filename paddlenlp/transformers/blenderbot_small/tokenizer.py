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
import shutil
from functools import lru_cache

import json
import re
from .. import PretrainedTokenizer
from paddle.utils import try_import

__all__ = ['BlenderbotSmallTokenizer']


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
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
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


class BlenderbotSmallTokenizer(PretrainedTokenizer):
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt"
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "blenderbot_small-90M":
                "https://paddlenlp.bj.bcebos.com/models/transformers/blenderbot_small/blenderbot_small-90M-vocab.json",
        },
        "merges_file": {
            "blenderbot_small-90M":
                "https://paddlenlp.bj.bcebos.com/models/transformers/blenderbot_small/blenderbot_small-90M-merges.txt",
        }
    }
    pretrained_init_configuration = {"blenderbot_small-90M": {}}

    def __init__(
            self,
            vocab_file,
            merges_file,
            errors='replace',
            max_len=None,
            special_tokens=None,
            bos_token="__start__",
            eos_token="__end__",
            unk_token="__unk__",
            pad_token="__null__",
            eol_token="__newln__",  # The token of newline.
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
        self.pat = r"\S+\n?"
        self.unk_id = self.encoder[unk_token]
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

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        token = re.sub("([.,!?()])", r" \1", token)
        token = re.sub("(')", r" \1 ", token)
        token = re.sub(r"\s{2,}", " ", token)
        if "\n" in token:
            token = token.replace("\n", " __newln__")

        tokens = token.split(" ")
        words = []
        for token in tokens:
            if not len(token):
                continue

            token = token.lower()
            word = tuple(token)
            word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
            pairs = get_pairs(word)

            if not pairs:
                words.append(token)
                continue

            while True:
                bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
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
                    except ValueError:
                        new_word.extend(word[i:])
                        break

                    if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
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
            word = "@@ ".join(word)
            word = word[:-4]

            self.cache[token] = word
            words.append(word)
        return " ".join(words)

    def tokenize(self, text):
        return self._tokenize(text)

    def _tokenize(self, text):
        """Split a string into tokens using BPE."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            bpe_tokens.extend([t for t in self.bpe(token).split(" ")])
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        if isinstance(tokens, str):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, self.unk_id)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, self.unk_id))
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

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens).replace("@@ ", "").strip()


    def convert_ids_to_string(self, ids):
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
