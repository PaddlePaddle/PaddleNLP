#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six
import os
import json
import regex as re
from functools import lru_cache
import numpy as np


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            try:
                return text.decode("utf-8", "ignore")
            except:
                return text.decode("gb18030", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            try:
                return text.decode("utf-8", "ignore")
            except:
                return text.decode("gb18030", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        items = convert_to_unicode(line.strip()).split("\t")
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def convert_tokens_to_ids(vocab, tokens):
    """convert tokens to vocab ids"""
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    """convert vocab ids to tokens"""
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


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
    bs = list(range(33, 126 + 1)) + list(range(161, 172 + 1)) + list(range(174, 255 + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
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


class Tokenizer(object):
    """RoBERTa Tokenizer"""
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.int_token = re.compile(r"^[0-9]+$")

    def bpe(self, token):
        """bpe tokenizing"""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
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
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """bpe encoding"""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """bpe decoding"""
        decoded_tokens = []
        for token in tokens:
            if self.int_token.match(str(token)):
                decoded_tokens.append(self.decoder[int(token)])
            else:
                decoded_tokens.append(str(token))
        text = ''.join(decoded_tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    def decode_token(self, token):
        if self.int_token.match(str(token)):
            token = self.decoder[int(token)]
        else:
            token = str(token)
        text = bytearray([self.byte_decoder[c] for c in token]).decode('utf-8', errors=self.errors)
        return text


class GptBpeTokenizer(object):
    """GptBpeTokenizer"""
    def __init__(self, vocab_file=None, encoder_json_file=None, vocab_bpe_file=None, do_lower_case=True):
        if vocab_file is None:
            vocab_file = "./model_files/dict/roberta_base_en.vocab.txt"
        if encoder_json_file is None:
            encoder_json_file = "./model_files/dict/roberta_base_en.encoder.json"
        if vocab_bpe_file is None:
            vocab_bpe_file = "./model_files/dict/roberta_base_en.vocab.bpe"

        with open(encoder_json_file, 'r') as f:
            encoder = json.load(f)
        with open(vocab_bpe_file, 'r', encoding="utf-8") as f:
            bpe_data = f.read()

        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        self.gptbpe_tokenizer = Tokenizer(encoder=encoder, bpe_merges=bpe_merges)
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.sep_token = '[SEP]'
        self.unk_token = '[UNK]'
        self.mask_token = '[MASK]'
        self.single_modal = 'madeupword0000'
        self.multi_modal = 'madeupword0001'

        self.cls_token_id = self.vocab['[CLS]']
        self.pad_token_id = self.vocab['[PAD]']
        self.sep_token_id = self.vocab['[SEP]']
        self.unk_token_id = self.vocab['[UNK]']
        self.mask_token_id = self.vocab['[MASK]']
        self.single_modal_id = self.vocab['madeupword0000']
        self.multi_modal_id = self.vocab['madeupword0001']

    def tokenize(self, text):
        """tokenize text to a list of tokens"""
        return [str(token) for token in self.gptbpe_tokenizer.encode(text)]

    def convert_tokens_to_ids(self, tokens):
        """convert tokens to vocab ids"""
        return convert_by_vocab(self.vocab, tokens)

    def convert_token_to_id(self, token):
        """convert token to vocab id"""
        return self.vocab[token]

    def convert_ids_to_tokens(self, ids):
        """convert vocab ids to tokens"""
        return convert_by_vocab(self.inv_vocab, ids)

    def convert_id_to_token(self, id):
        """convert vocab id to token"""
        return self.inv_vocab[id]

    def vocab_size(self):
        """get the vocab size"""
        return len(self.vocab)


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        """tokenize the text"""
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """convert tokens to vocab ids"""
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        """convert vocab ids to tokens"""
        return convert_by_vocab(self.inv_vocab, ids)

    def merge_subword(self, tokens):
        ret = []
        for token in tokens:
            if token.startswith("##"):
                real_token = token[2:]
                if len(ret):
                    ret[-1] += real_token
                else:
                    ret.append(real_token)
            else:
                ret.append(token)

        return ret


class CharTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        """tokenize the text"""
        split_tokens = []
        for token in text.lower().split(" "):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """convert tokens to vocab ids"""
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        """convert vocab ids to tokens"""
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
            do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer.

        Returns:
            A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
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


def test_gpt_bpe_tokenizer():
    """test the tokenizer"""
    tokenizer = GptBpeTokenizer()
    text = "Membership gives the ICC jurisdiction over alleged crimes committed " \
           "in Palestinian territories since last June . Israel and the United States " \
           "opposed the move , which could open the door to war crimes investigations against Israelis ."
    text = convert_to_unicode(text)
    tokens = tokenizer.tokenize(text)
    # tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.gptbpe_tokenizer.decode(tokens)
    tokens = tokenizer.gptbpe_tokenizer.encode(text)
    str_tokens = []
    for token in tokens:
        str_tokens.append(tokenizer.gptbpe_tokenizer.decode_token(token))
    tokens = tokenizer.gptbpe_tokenizer.decode(tokens)

def build_id2is_full_token(tokenizer, dtype="float32"):
    vocab_sz = tokenizer.vocab_size()
    is_full_token = [0.0] * vocab_sz
    token_strs = []

    for token_id in range(vocab_sz):
        token = tokenizer.convert_id_to_token(token_id)
        token_str = tokenizer.gptbpe_tokenizer.decode_token(token)
        token_strs.append(token_str)
        if token_str.startswith(' '):
            is_full_token[token_id] = 1.0

    is_full_token = np.array(is_full_token, dtype=dtype)
    print(is_full_token[500:520])
    print(token_strs[500:520])


if __name__ == "__main__":
    test_gpt_bpe_tokenizer()
    # tokenizer = GptBpeTokenizer()
    # build_id2is_full_token(tokenizer)
