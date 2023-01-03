# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The Open AI Team Authors and The HuggingFace Inc. team.
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

from paddle.utils import try_import
import os
import shutil
from .. import PretrainedTokenizer, AddedToken, BasicTokenizer
from ...utils.log import logger
from functools import lru_cache
import json

__all__ = ["CLIPTokenizer"]


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.
    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def whitespace_clean(text, re):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class CLIPTokenizer(PretrainedTokenizer):
    r"""
    Construct a CLIP tokenizer based on byte-level Byte-Pair-Encoding.

    This tokenizer inherits from :class:`~paddlenlp.transformers.gpt.tokenizer.GPTTokenizer`.
    For more information regarding those methods, please refer to this superclass.

    Args:
        vocab_file (str):
            Path to the vocabulary file.
            The vocab file contains a mapping from vocabulary strings to indices.
        merges_file (str):
            Path to the merge file.
            The merge file is used to split the input sentence into "subword" units.
            The vocab file is then used to encode those units as intices.
        errors (str):
            Paradigm to follow when decoding bytes to UTF-8.
            Defaults to `'replace'`.
        max_len (int, optional):
            The maximum value of the input sequence length.
            Defaults to `77`.
        bos_token (str, optional):
            The beginning of sequence token that was used during pretraining. Can be
            used a sequence classifier token.
            Defaults to `"<|startoftext|>"`.
        eos_token (str, optional):
            A special token representing the end of a sequence that was used during pretraining.
            Defaults to `"<|endoftext|>"`.
        unk_token (str, optional):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to `"<|endoftext|>"`.
        pad_token (str, optional):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to `"<|endoftext|>"`.

    Examples:
        .. code-block::

            from paddlenlp.transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32')
            print(tokenizer('He was a puppeteer'))

            '''
            {'input_ids': [49406, 797, 739, 320, 7116, 38820, 528, 49407]}
            '''

    """
    # merges and vocab same as GPT2
    resource_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "openai/clip-vit-base-patch32": "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-vit-base-patch32/vocab.json",
            "openai/clip-rn50": "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-rn50/vocab.json",
            "openai/clip-rn101": "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-rn101/vocab.json",
            "openai/clip-vit-large-patch14": "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-vit-large-patch14/vocab.json",
        },
        "merges_file": {
            "openai/clip-vit-base-patch32": "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-vit-base-patch32/merges.txt",
            "openai/clip-rn50": "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-rn50/merges.txt",
            "openai/clip-rn101": "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-rn101/merges.txt",
            "openai/clip-vit-large-patch14": "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-vit-large-patch14/merges.txt",
        },
    }
    pretrained_init_configuration = {
        "openai/clip-vit-base-patch32": {"max_len": 77},
        "openai/clip-rn50": {"max_len": 77},
        "openai/clip-rn101": {"max_len": 77},
        "openai/clip-vit-large-patch14": {"max_len": 77},
    }

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        max_len=77,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        **kwargs
    ):

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        self._build_special_tokens_map_extended(
            bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, pad_token=pad_token
        )

        try:
            import ftfy

            self.fix_text = ftfy.fix_text
        except ImportError:
            logger.warning("ftfy or spacy is not installed using BERT BasicTokenizer instead of ftfy.")
            self.nlp = BasicTokenizer(do_lower_case=True)
            self.fix_text = None
        self.re = try_import("regex")

        self._vocab_file = vocab_file
        self._merges_file = merges_file
        self.max_len = max_len if max_len is not None else int(1e12)

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}

        self.pat = self.re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            self.re.IGNORECASE,
        )

    @property
    def vocab_size(self):
        """
        Returns the size of vocabulary.

        Returns:
            int: The sum of size of vocabulary and the size of speical tokens.

        """
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens. A CLIP sequence has the following format:

        - single sequence: `<|startoftext|> X <|endoftext|>`

        Pairs of sequences are not the expected use case, but they will be handled without a separator.

        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs. Defaults to None.

        Returns:
            List[int]: List of input_id with the appropriate special tokens.
        """
        _bos = [self.bos_token_id]
        _eos = [self.eos_token_id]
        if token_ids_1 is None:
            return _bos + token_ids_0 + _eos
        return _bos + token_ids_0 + _eos + _eos + token_ids_1 + _eos

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is
        called when adding special tokens using the tokenizer ``encode`` methods.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.
        """
        eos = [self.eos_token_id]
        bos = [self.bos_token_id]

        if token_ids_1 is None:
            return len(bos + token_ids_0 + eos) * [0]
        return len(bos + token_ids_0 + eos + eos + token_ids_1 + eos) * [0]

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

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
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

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
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        if self.fix_text is None:
            text = " ".join(self.nlp.tokenize(text))
        else:
            text = whitespace_clean(self.fix_text(text), self.re).lower()

        for token in self.re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.decoder[index]

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string.
        """
        text = "".join(tokens)
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors=self.errors)
            .replace("</w>", " ")
            .strip()
        )
        return text

    def save_resources(self, save_directory):
        """
        Saves `SentencePiece <https://github.com/google/sentencepiece>`__ file
        (ends with '.spm') under `save_directory`.

        Args:
            save_directory (str): Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            source_path = getattr(self, "_%s" % name)

            save_path = os.path.join(save_directory, file_name)
            if os.path.abspath(source_path) != os.path.abspath(save_path):
                shutil.copyfile(source_path, save_path)

    def __call__(
        self,
        text,
        text_pair=None,
        max_length=None,
        stride=0,
        is_split_into_words=False,
        padding=False,
        truncation=False,
        return_position_ids=False,
        return_token_type_ids=False,  # don't return token_type_ids
        return_attention_mask=False,
        return_length=False,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_dict=True,
        return_offsets_mapping=False,
        add_special_tokens=True,
        pad_to_multiple_of=None,
        return_tensors=None,
        verbose: bool = True,
        **kwargs
    ):
        return super().__call__(
            text,
            text_pair,
            max_length,
            stride,
            is_split_into_words,
            padding,
            truncation,
            return_position_ids,
            return_token_type_ids,
            return_attention_mask,
            return_length,
            return_overflowing_tokens,
            return_special_tokens_mask,
            return_dict,
            return_offsets_mapping,
            add_special_tokens,
            pad_to_multiple_of,
            return_tensors,
            verbose,
            **kwargs,
        )
