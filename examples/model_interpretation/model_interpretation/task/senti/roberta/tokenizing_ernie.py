#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import six
import re
import logging
import tempfile
from pathlib import Path
from functools import partial
if six.PY2:
    from pathlib2 import Path
else:
    from pathlib import Path

from tqdm import tqdm
import numpy as np

import io

open = partial(io.open, encoding='utf8')

log = logging.getLogger(__name__)

_max_input_chars_per_word = 100


def _wordpiece(token, vocab, unk_token, prefix='##', sentencepiece_prefix=''):
    """ wordpiece: helloworld => [hello, ##world] """
    chars = list(token)
    if len(chars) > _max_input_chars_per_word:
        return [unk_token], [(0, len(chars))]

    is_bad = False
    start = 0
    sub_tokens = []
    sub_pos = []
    while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
            substr = "".join(chars[start:end])
            if start == 0:
                substr = sentencepiece_prefix + substr
            if start > 0:
                substr = prefix + substr
            if substr in vocab:
                cur_substr = substr
                break
            end -= 1
        if cur_substr is None:
            is_bad = True
            break
        sub_tokens.append(cur_substr)
        sub_pos.append((start, end))
        start = end
    if is_bad:
        return [unk_token], [(0, len(chars))]
    else:
        return sub_tokens, sub_pos

def _wordpiece_wo_unk(token, vocab, unk_token, prefix='##', sentencepiece_prefix=''):
    """ wordpiece: helloworld => [hello, ##world] """
    chars = list(token)
    if len(chars) > _max_input_chars_per_word:
        return [unk_token], [(0, len(chars))]

    is_bad = False
    start = 0
    sub_tokens = []
    sub_pos = []
    while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
            substr = "".join(chars[start:end])
            if start == 0:
                substr = sentencepiece_prefix + substr
            if start > 0:
                substr = prefix + substr
            if substr in vocab:
                cur_substr = substr
                break
            end -= 1
        if cur_substr is None:
            is_bad = True
            break
        sub_tokens.append(cur_substr)
        sub_pos.append((start, end))
        start = end
    if is_bad:
        assert len(chars) == 1
        return chars, [(0, len(chars))]
    else:
        return sub_tokens, sub_pos


class ErnieTokenizer(object):
    bce = 'https://ernie-github.cdn.bcebos.com/'
    resource_map = {
        'ernie-1.0': bce + 'model-ernie1.0.1.tar.gz',
        'ernie-2.0-en': bce + 'model-ernie2.0-en.1.tar.gz',
        'ernie-2.0-large-en': bce + 'model-ernie2.0-large-en.1.tar.gz',
        'ernie-tiny': bce + 'model-ernie_tiny.1.tar.gz',
        'ernie-gen-base-en': bce + 'model-ernie-gen-base-en.1.tar.gz',
        'ernie-gen-large-en': bce + 'model-ernie-gen-large-en.1.tar.gz',
        'ernie-gram-zh': bce + 'model-ernie-gram-zh.1.tar.gz',
        'ernie-gram-en': bce + 'model-ernie-gram-en.1.tar.gz',
        'ernie-2.0-large-ch': '/home/disk1/pengshuyuan/model-interp/ernie_large_ch/ernie_2.0_large_ch',
    }

    @classmethod
    def from_pretrained(cls,
                        pretrain_dir_or_url,
                        force_download=False,
                        **kwargs):
        if os.path.isfile(pretrain_dir_or_url):
            vocab_path = Path(pretrain_dir_or_url)
        else:
            if str(pretrain_dir_or_url) in cls.resource_map:
                url = cls.resource_map[str(pretrain_dir_or_url)]
                pretrain_dir = Path(url)
            else:
                log.info('pretrain dir %s not in %s, read from local' %
                        (pretrain_dir_or_url, repr(cls.resource_map)))
                pretrain_dir = Path(pretrain_dir_or_url)
            if not pretrain_dir.exists():
                raise ValueError('pretrain dir not found: %s, optional: %s' % (pretrain_dir, cls.resource_map.keys()))
            vocab_path = pretrain_dir / 'vocab.txt'
            if not vocab_path.exists():
                raise ValueError('no vocab file in pretrain dir: %s' %
                                pretrain_dir)
        vocab_dict = {
            j.strip().split('\t')[0]: i
            for i, j in enumerate(
                vocab_path.open(encoding='utf8').readlines())
        }
        t = cls(vocab_dict, **kwargs)
        return t

    def __init__(self,
                 vocab,
                 unk_token='[UNK]',
                 sep_token='[SEP]',
                 cls_token='[CLS]',
                 pad_token='[PAD]',
                 mask_token='[MASK]',
                 wordpiece_prefix='##',
                 sentencepiece_prefix='',
                 lower=True,
                 encoding='utf8',
                 special_token_list=[]):
        if not isinstance(vocab, dict):
            raise ValueError('expect `vocab` to be instance of dict, got %s' %
                             type(vocab))
        self.vocab = vocab
        self.inverse_vocab = [item[0] for item in sorted(self.vocab.items(), key=lambda x: x[1])]
        self.lower = lower
        self.prefix = wordpiece_prefix
        self.sentencepiece_prefix = sentencepiece_prefix
        self.pad_id = self.vocab[pad_token]
        self.cls_id = cls_token and self.vocab[cls_token]
        self.sep_id = sep_token and self.vocab[sep_token]
        self.unk_id = unk_token and self.vocab[unk_token]
        self.mask_id = mask_token and self.vocab[mask_token]
        self.unk_token = unk_token
        special_tokens = {
            pad_token, cls_token, sep_token, unk_token, mask_token
        } | set(special_token_list)
        pat_str = ''
        for t in special_tokens:
            if t is None:
                continue
            pat_str += '(%s)|' % re.escape(t)
        pat_str += r'([a-zA-Z0-9]+|\S)'
        log.debug('regex: %s' % pat_str)
        self.pat = re.compile(pat_str)
        self.encoding = encoding

    def tokenize(self, text, wo_unk=False):
        if len(text) == 0:
            return []
        if six.PY3 and not isinstance(text, six.string_types):
            text = text.decode(self.encoding)
        if six.PY2 and isinstance(text, str):
            text = text.decode(self.encoding)

        res = []
        for match in self.pat.finditer(text):
            match_group = match.group(0)
            if match.groups()[-1]:
                if self.lower:
                    match_group = match_group.lower()
                if wo_unk:
                    wordpiece = _wordpiece_wo_unk
                else:
                    wordpiece = _wordpiece
                words, _ = wordpiece(
                    match_group,
                    vocab=self.vocab,
                    unk_token=self.unk_token,
                    prefix=self.prefix,
                    sentencepiece_prefix=self.sentencepiece_prefix)
            else:
                words = [match_group]
            res += words
        return res

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    def truncate(self, id1, id2, seqlen):
        len1 = len(id1)
        len2 = len(id2)
        half = seqlen // 2
        if len1 > len2:
            len1_truncated, len2_truncated = max(half, seqlen - len2), min(
                half, len2)
        else:
            len1_truncated, len2_truncated = min(half, seqlen - len1), max(
                half, seqlen - len1)
        return id1[:len1_truncated], id2[:len2_truncated]

    def build_for_ernie(self, text_id, pair_id=[]):
        """build sentence type id, add [CLS] [SEP]"""
        text_id_type = np.zeros_like(text_id, dtype=np.int64)
        ret_id = np.concatenate([[self.cls_id], text_id, [self.sep_id]], 0)
        ret_id_type = np.concatenate([[0], text_id_type, [0]], 0)

        if len(pair_id):
            pair_id_type = np.ones_like(pair_id, dtype=np.int64)
            ret_id = np.concatenate([ret_id, pair_id, [self.sep_id]], 0)
            ret_id_type = np.concatenate([ret_id_type, pair_id_type, [1]], 0)
        return ret_id, ret_id_type

    def encode(self, text, pair=None, truncate_to=None):
        text_id = np.array(
            self.convert_tokens_to_ids(self.tokenize(text)), dtype=np.int64)
        text_id_type = np.zeros_like(text_id, dtype=np.int64)
        if pair is not None:
            pair_id = np.array(
                self.convert_tokens_to_ids(self.tokenize(pair)),
                dtype=np.int64)
        else:
            pair_id = []
        if truncate_to is not None:
            text_id, pair_id = self.truncate(text_id, [] if pair_id is None
                                             else pair_id, truncate_to)

        ret_id, ret_id_type = self.build_for_ernie(text_id, pair_id)
        return ret_id, ret_id_type
