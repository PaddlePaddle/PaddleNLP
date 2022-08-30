# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from faster_tokenizer import Tokenizer

__all__ = ['BaseFasterTokenizer']


class BaseFasterTokenizer:

    def __init__(self, tokenizer_impl, parma_dict=None):
        self._tokenizer = tokenizer_impl
        self._parma_dict = parma_dict if parma_dict is not None else {}

    def __repr__(self):
        return "Tokenizer(vocabulary_size={}, {})".format(
            self._tokenizer.get_vocab_size(),
            ", ".join(k + "=" + str(v) for k, v in self._parma_dict.items()),
        )

    def num_special_tokens_to_add(self, is_pair):
        return self._tokenizer.num_special_tokens_to_add(is_pair)

    def get_vocab(self, with_added_tokens=True):
        return self._tokenizer.get_vocab(with_added_tokens=with_added_tokens)

    def get_vocab_size(self, with_added_tokens=True):
        return self._tokenizer.get_vocab_size(
            with_added_tokens=with_added_tokens)

    def enable_padding(
        self,
        direction="right",
        pad_id=0,
        pad_type_id=0,
        pad_token="[PAD]",
        pad_to_multiple_of=None,
        length=None,
    ):
        return self._tokenizer.enable_padding(
            direction=direction,
            pad_to_multiple_of=pad_to_multiple_of,
            pad_id=pad_id,
            pad_type_id=pad_type_id,
            pad_token=pad_token,
            length=length,
        )

    def disable_padding(self):
        self._tokenizer.disable_padding()

    @property
    def padding(self):
        return self._tokenizer.padding

    def enable_truncation(self,
                          max_length,
                          stride=0,
                          strategy="longest_first",
                          direction="right"):
        self._tokenizer.enable_truncation(max_length, stride, strategy,
                                          direction)

    def disable_truncation(self):
        self._tokenizer.disable_truncation()

    def truncation(self):
        return self._tokenizer.truncation

    def add_tokens(self, tokens):
        return self._tokenizer.add_tokens(tokens)

    def add_special_tokens(self, special_tokens):
        return self._tokenizer.add_special_tokens(tokens)

    def encode(
        self,
        sequence,
        pair=None,
        is_pretokenized=False,
        add_special_tokens=True,
    ):
        if sequence is None:
            raise ValueError("encode: `sequence` can't be `None`")
        return self._tokenizer.encode(sequence, pair, is_pretokenized,
                                      add_special_tokens)

    def encode_batch(self,
                     inputs,
                     add_special_tokens=True,
                     is_pretokenized=False):
        if inputs is None:
            raise ValueError("encode_batch: `inputs` can't be `None`")
        return self._tokenizer.encode_batch(inputs, add_special_tokens,
                                            is_pretokenized)

    def decode(self, ids, skip_special_tokens=True) -> str:
        if ids is None:
            raise ValueError(
                "None input is not valid. Should be a list of integers.")

        return self._tokenizer.decode(ids,
                                      skip_special_tokens=skip_special_tokens)

    def decode_batch(self, sequences, skip_special_tokens=True) -> str:
        if sequences is None:
            raise ValueError(
                "None input is not valid. Should be list of list of integers.")

        return self._tokenizer.decode_batch(
            sequences, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token):
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id):
        return self._tokenizer.id_to_token(id)

    def post_process(self, encoding, pair=None, add_special_tokens=True):
        return self._tokenizer.post_process(encoding, pair, add_special_tokens)

    @property
    def model(self):
        return self._tokenizer.model

    @model.setter
    def model(self, model):
        self._tokenizer.model = model

    @property
    def normalizer(self):
        return self._tokenizer.normalizer

    @normalizer.setter
    def normalizer(self, normalizer):
        self._tokenizer.normalizer = normalizer

    @property
    def pretokenizer(self):
        return self._tokenizer.pretokenizer

    @pretokenizer.setter
    def pretokenizer(self, pretokenizer):
        self._tokenizer.pretokenizer = pretokenizer

    @property
    def postprocessor(self):
        return self._tokenizer.postprocessor

    @postprocessor.setter
    def postprocessor(self, postprocessor):
        self._tokenizer.postprocessor = postprocessor

    @property
    def decoder(self):
        return self._tokenizer.decoder

    @decoder.setter
    def decoder(self, decoder):
        self._tokenizer.decoder = decoder

    def save(self, path, pretty=True):
        self._tokenizer.save(path, pretty)

    def to_str(self, pretty=True):
        return self._tokenizer.to_str(pretty)

    @staticmethod
    def from_str(json):
        return Tokenizer.from_str(json)

    @staticmethod
    def from_file(path):
        return Tokenizer.from_file(path)
