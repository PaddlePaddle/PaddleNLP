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
import pickle
import shutil
import json
import regex as re
from nltk import tokenize
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

from paddle.utils import try_import
from paddlenlp.utils.env import MODEL_HOME
from .. import PretrainedTokenizer
from ..ernie.tokenizer import ErnieTokenizer

__all__ = ['ErnieDocTokenizer', 'BPETokenizer']


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


class ErnieDocTokenizer(ErnieTokenizer):
    r"""
    Constructs an ERNIE-Doc tokenizer. It uses a basic tokenizer to do punctuation
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
            from paddlenlp.transformers import ErnieDocTokenizer
            tokenizer = ErnieDocTokenizer.from_pretrained('ernie-doc-zh')
            encoded_inputs = tokenizer('这是一个测试样例')
            # encoded_inputs: 
            # { 
            #   'input_ids': [1, 47, 10, 7, 27, 558, 525, 314, 656, 2], 
            #   'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # }


    """
    # TODO(zhoushunjie): need to add vocab.txt url
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {"vocab_file": {}}
    pretrained_init_configuration = {"": {"": True}, }

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]"):
        super(ErnieDocTokenizer, self).__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token)


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

    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]

    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1

    cs = [chr(n) for n in cs]

    ddict = dict(zip(bs, cs))
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


class Encoder(object):
    def __init__(self,
                 encoder,
                 bpe_merges,
                 errors='replace',
                 special_tokens=["[SEP]", "[p]", "[q]", "[/q]"]):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        # print('111',self.byte_encoder)
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.re = re
        self.special_tokens = special_tokens

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = self.re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

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
        tokens = text.split(' ')
        sub_tokens = []
        for token_i, token in enumerate(tokens):
            if self.is_special_token(token):
                if token_i == 0:
                    sub_tokens.extend([token])
                else:
                    sub_tokens.extend([" " + token])
            else:
                if token_i == 0:
                    sub_tokens.extend(self.re.findall(self.pat, token))
                else:
                    sub_tokens.extend(self.re.findall(self.pat, " " + token))
        return sub_tokens

    def tokenize_old(self, text):
        return self.re.findall(self.pat, text)

    def is_special_token(self, tok):
        if isinstance(tok, int):
            return False
        res = False
        for t in self.special_tokens:
            # if tok.find(t) != -1:
            if tok.strip() == t:
                res = True
                break
        return res

    def tokenize_bpe(self, token):

        if self.is_special_token(token):
            return [token.strip()]  # remove space for convert_to_ids
        else:
            if six.PY2:
                token = ''.join(self.byte_encoder[ord(b)]
                                for b in token.encode('utf-8'))
            else:
                token = ''.join(self.byte_encoder[b]
                                for b in token.encode('utf-8'))
            return [
                self.encoder[bpe_token]
                for bpe_token in self.bpe(token).split(' ')
            ]

    def encode(self, text):
        bpe_tokens = []
        for token in self.tokenize(text):
            bpe_tokens.extend(self.tokenize_bpe(token))
        return bpe_tokens

    def decode(self, tokens):
        pre_token_i = 0
        texts = []
        for token_i, token in enumerate(tokens):
            if self.is_special_token(token):
                # proprecess tokens before token_i
                if token_i - pre_token_i > 0:
                    text = ''.join([
                        self.decoder[int(tok)]
                        for tok in tokens[pre_token_i:token_i]
                    ])
                    text = bytearray(
                        [self.byte_decoder[c] for c in text]).decode(
                            'utf-8', errors=self.errors)
                    texts.append(text)
                # texts.append(token)
                if token_i == 0:
                    texts.append(
                        token
                    )  # in the beginning, there is no space before special tokens
                else:
                    texts.extend(
                        [" ", token]
                    )  # in middle sentence, there must be a space before special tokens
                pre_token_i = token_i + 1

        if pre_token_i < len(tokens):
            text = ''.join(
                [self.decoder[int(tok)] for tok in tokens[pre_token_i:]])
            text = bytearray([self.byte_decoder[c] for c in text]).decode(
                'utf-8', errors=self.errors)
            texts.append(text)

        return ''.join(texts)


def get_encoder(encoder_json_path, vocab_bpe_path):
    with open(encoder_json_path, 'r') as f:
        encoder = json.load(f)
    with open(vocab_bpe_path, 'r') as f:
        bpe_data = f.read()
    bpe_merges = [
        tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]
    ]

    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges, )


class BPETokenizer(PretrainedTokenizer):
    """ Runs bpe tokenize """

    def __init__(self,
                 vocab_file,
                 encoder_json_path="./configs/encoder.json",
                 vocab_bpe_path="./configs/vocab.bpe",
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 params=None):
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
        self.encoder = get_encoder(encoder_json_path, vocab_bpe_path)

    def tokenize(self, text, is_sentencepiece=True):
        text = convert_to_unicode(text)
        text = " ".join(text.split())  # remove duplicate whitespace
        if is_sentencepiece:
            sents = tokenize.sent_tokenize(text)
            bpe_ids = sum([self.encoder.encode(sent) for sent in sents], [])
        else:
            bpe_ids = self.encoder.encode(text)
        tokens = [str(bpe_id) for bpe_id in bpe_ids]
        return tokens
