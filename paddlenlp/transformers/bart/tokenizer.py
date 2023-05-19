# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

import json
import os
import shutil
from functools import lru_cache

from paddle.utils import try_import

from .. import AddedToken, PretrainedTokenizer

__all__ = ["BartTokenizer"]

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bart-base": 1024,
    "bart-large": 1024,
}


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


class BartTokenizer(PretrainedTokenizer):
    r"""
    Construct a BART tokenizer based on byte-level Byte-Pair-Encoding.

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
            Defaults to `None`.
        bos_token (str, optional):
            The beginning of sequence token that was used during pretraining. Can be
            used a sequence classifier token.
            Defaults to `"<s>"`.
        eos_token (str, optional):
            A special token representing the end of a sequence that was used during pretraining.
            Defaults to `"</s>"`.
        cls_token (str, optional):
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens.
            Defaults to `"<s>"`.
        sep_token (str, optional):
            A special token separating two different sentences in the same input.
            Defaults to `"</s>"`.
        unk_token (str, optional):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to `"<unk>"`.
        pad_token (str, optional):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to `"<pad>"`.
        mask_token (str, optional):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to `"<mask>"`.

    Examples:
        .. code-block::

            from paddlenlp.transformers import BartTokenizer

            tokenizer = BartTokenizer.from_pretrained('bart-base')
            print(tokenizer('He was a puppeteer'))

            '''
            {'input_ids': [0, 894, 21, 10, 32986, 9306, 254, 2],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
            '''

    """
    # merges and vocab same as GPT2
    resource_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "bart-base": "https://bj.bcebos.com/paddlenlp/models/transformers/bart/bart-base-vocab.json",
            "bart-large": "https://bj.bcebos.com/paddlenlp/models/transformers/bart/bart-large-vocab.json",
        },
        "merges_file": {
            "bart-base": "https://bj.bcebos.com/paddlenlp/models/transformers/bart/bart-base-merges.txt",
            "bart-large": "https://bj.bcebos.com/paddlenlp/models/transformers/bart/bart-large-merges.txt",
        },
    }
    pretrained_init_configuration = {"bart-base": {}, "bart-large": {}}
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        self._build_special_tokens_map_extended(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
        )

        self._vocab_file = vocab_file
        self._merges_file = merges_file
        self.num_command_tokens = 2
        self.num_type_tokens = 2

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)

        self.decoder = {v: k for k, v in self.encoder.items()}

        self.num_tokens = len(self.encoder)
        self.num_text_tokens = self.num_tokens - 1
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with open(merges_file, encoding="utf-8") as f:
            bpe_data = f.read().split("\n")[1:-1]

        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        re = try_import("regex")
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def _bpe_encode(self, text):
        bpe_tokens = []
        re = try_import("regex")
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification
        tasks by concatenating and adding special tokens.
        """
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        if token_ids_1 is None:
            return _cls + token_ids_0 + _sep
        return _cls + token_ids_0 + _sep + _sep + token_ids_1 + _sep

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
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    @property
    def vocab_size(self):
        """
        Returns the size of vocabulary.

        Returns:
            int: The sum of size of vocabulary and the size of speical tokens.

        """

        return len(self.encoder)

    @property
    def eol_token_id(self):
        if self.eol_token is None:
            return None
        return self.convert_tokens_to_ids(self.eol_token)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

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
                except:  # noqa: E722
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
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        re = try_import("regex")
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):

        return self.decoder[index]

    def convert_ids_to_string(self, ids):
        """
        Converts a single index or a sequence of indices to texts.

        Args:
            ids (int|List[int]):
                The token id (or token ids) to be converted to text.

        Returns:
            str: The decoded text.

        Example:
            .. code-block::

                from paddlenlp.transformers import GPTTokenizer
                tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')
                print(tokenizer.convert_ids_to_string(tokenizer.convert_ids_to_string([14618, 284, 779, 350, 37382, 47, 37382, 290, 350, 37382, 45, 19930]))
                # 'Welcome to use PaddlePaddle and PaddleNLP'

        """

        text = "".join([self.decoder[id] for id in ids])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
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

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string.
        """
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def build_offset_mapping_with_special_tokens(self, offset_mapping_0, offset_mapping_1=None):
        """
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens.

        A BERT offset_mapping has the following format:

        - single sequence:      ``(0,0) X (0,0)``
        - pair of sequences:        ``(0,0) A (0,0) B (0,0)``

        Args:
            offset_mapping_ids_0 (List[tuple]):
                List of wordpiece offsets to which the special tokens will be added.
            offset_mapping_ids_1 (List[tuple], optional):
                Optional second list of wordpiece offsets for offset mapping pairs. Defaults to None.

        Returns:
            List[tuple]: A list of wordpiece offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0), (0, 0)] + offset_mapping_1 + [(0, 0)]
