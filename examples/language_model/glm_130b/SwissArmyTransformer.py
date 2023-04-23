# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Union

import sentencepiece_model_pb2 as sp_model
from icetk import TextTokenizer, auto_create

logger = logging.getLogger(__name__)


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError("detokenizer is not implemented for {} " "tokenizer".format(self.name))

    @property
    def cls(self):
        raise NotImplementedError("CLS is not provided for {} " "tokenizer".format(self.name))

    @property
    def sep(self):
        raise NotImplementedError("SEP is not provided for {} " "tokenizer".format(self.name))

    @property
    def pad(self):
        raise NotImplementedError("PAD is not provided for {} " "tokenizer".format(self.name))

    @property
    def eod(self):
        raise NotImplementedError("EOD is not provided for {} " "tokenizer".format(self.name))

    @property
    def mask(self):
        raise NotImplementedError("MASK is not provided for {} " "tokenizer".format(self.name))


class GLM130BTokenizer:
    def __init__(
        self,
        path="~/.icetk_models",
        max_blank_length=80,
        byte_fallback=True,
    ):
        if path is not None:
            self.path = os.path.expanduser(path)
        self.special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "<eod>", "<sop>", "<eop>", "<ENC>", "<dBLOCK>"]
        self.max_blank_length = max_blank_length
        self.byte_fallback = byte_fallback

    @staticmethod
    def _configure_tokenizer(
        text_tokenizer: TextTokenizer,
        special_tokens: List[str],
        max_blank_length: int,
        byte_fallback: bool,
        encode_special_tokens=False,
    ):
        # special token
        special_token_type = 4 if encode_special_tokens else 3  # 3 - CONTROL, 4 - USER_DEFINE
        for token in special_tokens:
            text_tokenizer.proto.pieces.append(
                sp_model.ModelProto.SentencePiece(piece=token, score=0.0, type=special_token_type)
            )
        # whitespaces
        for token in [GLM130BTokenizer.get_tab_token()] + [
            GLM130BTokenizer.get_blank_token(i) for i in range(2, max_blank_length + 1)
        ]:
            text_tokenizer.proto.pieces.append(sp_model.ModelProto.SentencePiece(piece=token, score=0.0, type=4))
        # byte fallback
        if byte_fallback:
            text_tokenizer.proto.trainer_spec.byte_fallback = True
            for i in range(256):
                text_tokenizer.proto.pieces.append(
                    sp_model.ModelProto.SentencePiece(piece="<0x{:02X}>".format(i), score=0.0, type=6)
                )
        text_tokenizer.refresh()

    def _get_text_tokenizer(self, encode_special_tokens=False):
        name = "_special_text_tokenizer" if encode_special_tokens else "_text_tokenizer"
        if not hasattr(self, name):
            fp = os.path.join(self.path, "ice_text.model")
            auto_create(fp)
            tokenizer = TextTokenizer(fp)
            self._configure_tokenizer(
                tokenizer, self.special_tokens, self.max_blank_length, self.byte_fallback, encode_special_tokens
            )
            setattr(self, name, tokenizer)
        return getattr(self, name)

    @staticmethod
    def get_blank_token(length: int):
        assert length >= 2
        return f"<|blank_{length}|>"

    @staticmethod
    def get_tab_token():
        return "<|tab|>"

    @property
    def text_tokenizer(self):
        return self._get_text_tokenizer(encode_special_tokens=False)

    @property
    def special_text_tokenizer(self):
        return self._get_text_tokenizer(encode_special_tokens=True)

    @property
    def num_image_tokens(self):
        return 20000

    @property
    def num_text_tokens(self):
        return (
            self.text_tokenizer.num_tokens
            + len(self.special_tokens)
            + (self.max_blank_length - 2)
            + (256 if self.byte_fallback else 0)
        )

    @property
    def num_tokens(self):
        return self.num_image_tokens + self.num_text_tokens

    @staticmethod
    def _encode_whitespaces(text: str, max_len: int = 80):
        text = text.replace("\t", GLM130BTokenizer.get_tab_token())
        for i in range(max_len, 1, -1):
            text = text.replace(" " * i, GLM130BTokenizer.get_blank_token(i))
        return text

    def _preprocess(self, text: str, linebreak=True, whitespaces=True):
        if linebreak:
            text = text.replace("\n", "<n>")
        if whitespaces:
            text = self._encode_whitespaces(text, max_len=self.max_blank_length)
        return text

    def encode(
        self, text: str, linebreak=True, whitespaces=True, special_tokens=False, add_dummy_prefix=True
    ) -> List[int]:
        """
        @param text: Text to encode.
        @param linebreak: Whether to encode newline (\n) in text.
        @param whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        @param special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        @param add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)
        if not add_dummy_prefix:
            text = "<n>" + text
        tmp = self._get_text_tokenizer(encode_special_tokens=special_tokens).encode(text)
        tokens = [x + self.num_image_tokens for x in tmp]
        return tokens if add_dummy_prefix else tokens[2:]

    def decode(self, text_ids: List[int], special_tokens=False) -> str:
        ids = [int(_id) - self.num_image_tokens for _id in text_ids]
        text = self._get_text_tokenizer(encode_special_tokens=special_tokens).decode(ids)
        text = text.replace("<n>", "\n")
        text = text.replace(GLM130BTokenizer.get_tab_token(), "\t")
        for i in range(2, self.max_blank_length + 1):
            text = text.replace(self.get_blank_token(i), " " * i)
        return text

    def tokenize(
        self, text: str, linebreak=True, whitespaces=True, special_tokens=False, add_dummy_prefix=True
    ) -> List[str]:
        """
        @param text: Text to encode.
        @param linebreak: Whether to encode newline (\n) in text.
        @param whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        @param special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        @param add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)
        if not add_dummy_prefix:
            text = "<n>" + text
        tokens = self._get_text_tokenizer(encode_special_tokens=special_tokens).tokenize(text)
        return tokens if add_dummy_prefix else tokens[2:]

    def __getitem__(self, x: Union[int, str]):
        if isinstance(x, int):
            if x < self.num_image_tokens:
                return "<image_{}>".format(x)
            else:
                return self.text_tokenizer.convert_id_to_token(x - self.num_image_tokens)
        elif isinstance(x, str):
            if x.startswith("<image_") and x.endswith(">") and x[7:-1].isdigit():
                return int(x[7:-1])
            else:
                return self.text_tokenizer.convert_token_to_id(x) + self.num_image_tokens
        else:
            raise ValueError("The key should be str or int.")


class _IceTokenizer(AbstractTokenizer):
    """Hardcoded tokenizer."""

    def __init__(self, max_blank_len=80):
        name = "IceTokenizer"
        super().__init__(name)

        self.tokenizer = GLM130BTokenizer()
        self.num_tokens = 150000
        self.add_special_tokens(
            ["[MASK]", "[gMASK]", "[sMASK]", "eod", "sop", "eop", "ENC", "dBLOCK"]
            + ["<t>"]
            + [f"<blank_{i}>" for i in range(2, max_blank_len + 1)]
        )

        self.sentence_end_decoder = {
            20007: ".",
            20031: "？",
            20035: "！",
            20027: "；",
            20012: ":",
            83823: "。",
            145670: "…",
        }

        self.special_tokens["eos"] = 20002
        self.special_tokens_decoder[20002] = "</s>"

    def add_special_tokens(self, special_tokens):
        """Add a list of additional tokens to the encoder.
        The additional tokens are indexed starting from the last index of the
        current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, self.num_tokens + i) for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}
        # for k, v in self.special_tokens.items():
        #     self.tokenizer.decoder[v] = "\u0120" + k
        logger.info("Special tokens {}".format(self.special_tokens))

    def get_command(self, token):
        return self.special_tokens[token]

    def contains_sentence_end(self, idx):
        return idx in self.sentence_end_decoder

    def IdToToken(self, idx):
        if idx == 0:
            return "[pad]"
        elif idx in self.special_tokens_decoder:
            return f"[{self.special_tokens_decoder[idx]}]"
        else:
            return self.tokenizer.decode([idx])

    def TokenToId(self, token):
        if token == "[pad]":
            return 0
        elif token in self.special_tokens:
            return self.special_tokens[token]
        else:
            return self.tokenizer.encode(token)[0]

    @property
    def vocab_size(self):
        return self.tokenizer.num_tokens

    @property
    def vocab(self):
        assert False
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        assert False
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        split = [-1]
        for i, token in enumerate(token_ids):
            if token in self.special_tokens_decoder:
                split.append(i)
        split.append(len(token_ids))
        text = ""
        for i in range(len(split) - 1):
            if i > 0:
                text += self.IdToToken(token_ids[split[i]])
            text += self.tokenizer.decode(token_ids[split[i] + 1 : split[i + 1]])
        return text

    @property
    def eod(self):
        return self.get_special_token("eod")


def get_tokenizer(args=None, *, tokenizer_type=None):
    get_tokenizer.tokenizer_type = tokenizer_type
    get_tokenizer.tokenizer = _IceTokenizer()

    return get_tokenizer.tokenizer
