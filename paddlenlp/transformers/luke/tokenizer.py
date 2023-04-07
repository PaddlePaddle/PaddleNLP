# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""Tokenization classes for LUKE."""

from typing import Dict, List, Optional, Union

try:
    import regex as re
except:
    import re

import itertools
import json
import sys
import warnings
from itertools import repeat

from .. import RobertaBPETokenizer

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func


__all__ = ["LukeTokenizer"]
_add_prefix_space = False

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"luke-base": 514, "luke-large": 514}


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


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
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


class LukeTokenizer(RobertaBPETokenizer):
    """
    Constructs a Luke tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        vocab_file (str):
            The vocabulary file path (ends with '.json') required to instantiate
            a `WordpieceTokenizer`.
        entity_file (str):
            The entity vocabulary file path (ends with '.tsv') required to instantiate
            a `EntityTokenizer`.
        do_lower_case (bool):
            Whether or not to lowercase the input when tokenizing.
            Defaults to`True`.
        unk_token (str):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "[UNK]".
        sep_token (str):
            A special token separating two different sentences in the same input.
            Defaults to "[SEP]".
        pad_token (str):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "[PAD]".
        cls_token (str):
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to "[CLS]".
        mask_token (str):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to "[MASK]".

    Examples:
        .. code-block::

            from paddlenlp.transformers import LukeTokenizer
            tokenizer = LukeTokenizer.from_pretrained('luke-large)

            tokens = tokenizer('Beyoncé lives in Los Angeles', entity_spans=[(0, 7), (17, 28)])
            #{'input_ids': [0, 40401, 261, 12695, 1074, 11, 1287, 1422, 2], 'entity_ids': [1657, 32]}

    """

    # resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "entity_file": "entity_vocab.json",
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "luke-base": "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-base/vocab.json",
            "luke-large": "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-large/vocab.json",
        },
        "merges_file": {
            "luke-base": "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-base/merges.txt",
            "luke-large": "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-large/merges.txt",
        },
        "entity_file": {
            "luke-base": "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-base/entity_vocab.json",
            "luke-large": "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-large/entity_vocab.json",
        },
    }
    pretrained_init_configuration = {"luke-base": {"do_lower_case": True}, "luke-large": {"do_lower_case": True}}

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        entity_file,
        merges_file,
        do_lower_case=True,
        unk_token="<unk>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="<s>",
        mask_token="<mask>",
        **kwargs
    ):

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        with open(entity_file, encoding="utf-8") as entity_vocab_handle:
            self.entity_vocab = json.load(entity_vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self._all_special_tokens = [unk_token, sep_token, pad_token, cls_token, mask_token]
        self.errors = "replace"  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # RobertaTokenizer don't maintain the entity_file resource file name,
        # so we should not set it as a param in super.__init__ function
        self._entity_file = entity_file
        super(LukeTokenizer, self).__init__(
            vocab_file,
            merges_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def sep_token_id(self):
        return self.encoder[self.sep_token]

    @property
    def cls_token_id(self):
        return self.encoder[self.cls_token]

    @property
    def pad_token_id(self):
        return self.encoder[self.pad_token]

    @property
    def unk_token_id(self):
        return self.encoder[self.unk_token]

    def get_entity_vocab(self):
        """Get the entity vocab"""
        return self.entity_vocab

    def _convert_token_to_id(self, token):
        """Converts a token (str/unicode) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.decoder.get(index)

    def _tokenize(self, text, add_prefix_space=False):
        if add_prefix_space:
            text = " " + text

        bpe_tokens = []
        for token in re.findall(self.pat, text):
            if sys.version_info[0] == 2:
                token = "".join(
                    self.byte_encoder[ord(b)] for b in token
                )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            else:
                token = "".join(
                    self.byte_encoder[b] for b in token.encode("utf-8")
                )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def __call__(
        self,
        text,
        text_pair=None,
        entity_spans=None,
        entity_spans_pair=None,
        entities=None,
        entities_pair=None,
        max_mention_length=30,
        max_length: Optional[int] = None,
        stride=0,
        add_prefix_space=False,
        is_split_into_words=False,
        padding=False,
        truncation="longest_first",
        return_position_ids=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        **kwargs
    ):
        """
        Performs tokenization and uses the tokenized tokens to prepare model
        inputs. It supports sequence or sequence pair as input, and batch input
        is allowed. `self.encode()` or `self.batch_encode()` would be called
        separately for single or batch input depending on input format and
        `is_split_into_words` argument.

        Args:
            text (str, List[str] or List[List[str]]):
                The sequence or batch of sequences to be processed. One sequence
                is a string or a list of strings depending on whether it has been
                pretokenized. If each sequence is provided as a list of strings
                (pretokenized), you must set `is_split_into_words` as `True` to
                disambiguate with a batch of sequences.
            text_pair (str, List[str] or List[List[str]], optional):
                Same as `text` argument, while it represents for the latter
                sequence of the sequence pair.
            entity_spans (`List[Tuple[int, int]]`, `List[List[Tuple[int, int]]]`, *optional*):
                The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
                with two integers denoting character-based(different from transformers LUKE) start and end positions
                of entities. If you specify `"entity_classification"` or `"entity_pair_classification"` as the `task`
                argument in the constructor, the length of each sequence must be 1 or 2, respectively. If you specify
                `entities`, the length of each sequence must be equal to the length of each sequence of `entities`.
            entity_spans_pair (`List[Tuple[int, int]]`, `List[List[Tuple[int, int]]]`, *optional*):
                The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
                with two integers denoting character-based start and end positions of entities. If you specify the
                `task` argument in the constructor, this argument is ignored. If you specify `entities_pair`, the
                length of each sequence must be equal to the length of each sequence of `entities_pair`.
            entities (`List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
                representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
                Angeles). This argument is ignored if you specify the `task` argument in the constructor. The length of
                each sequence must be equal to the length of each sequence of `entity_spans`. If you specify
                `entity_spans` without specifying this argument, the entity sequence or the batch of entity sequences
                is automatically constructed by filling it with the [MASK] entity.
            entities_pair (`List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
                representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
                Angeles). This argument is ignored if you specify the `task` argument in the constructor. The length of
                each sequence must be equal to the length of each sequence of `entity_spans_pair`. If you specify
                `entity_spans_pair` without specifying this argument, the entity sequence or the batch of entity
                sequences is automatically constructed by filling it with the [MASK] entity.
            max_mention_length (`int`):
                The entity_position_ids's length.
            max_length (int, optional):
                If set to a number, will limit the total sequence returned so
                that it has a maximum length. If there are overflowing tokens,
                those overflowing tokens will be added to the returned dictionary
                when `return_overflowing_tokens` is `True`. Defaults to `None`.
            stride (int, optional):
                Only available for batch input of sequence pair and mainly for
                question answering usage. When for QA, `text` represents questions
                and `text_pair` represents contexts. If `stride` is set to a
                positive number, the context will be split into multiple spans
                where `stride` defines the number of (tokenized) tokens to skip
                from the start of one span to get the next span, thus will produce
                a bigger batch than inputs to include all spans. Moreover, 'overflow_to_sample'
                and 'offset_mapping' preserving the original example and position
                information will be added to the returned dictionary. Defaults to 0.
            add_prefix_space (bool, optional):
                The tokenizer will add a space at the beginning of the sentence when it set to `True`.
                Defaults to `False`.
            padding (bool, optional):
                If set to `True`, the returned sequences would be padded up to
                `max_length` specified length according to padding side
                (`self.padding_side`) and padding token id. Defaults to `False`.
            truncation (str, optional):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence
                until the input is under `max_length` starting from the longest
                one at each token (when there is a pair of input sequences).
                - 'only_first': Only truncate the first sequence.
                - 'only_second': Only truncate the second sequence.
                - 'do_not_truncate': Do not truncate (raise an error if the input
                sequence is longer than `max_length`).

                Defaults to 'longest_first'.
            return_position_ids (bool, optional):
                Whether to include tokens position ids in the returned dictionary.
                Defaults to `False`.
            return_token_type_ids (bool, optional):
                Whether to include token type ids in the returned dictionary.
                Defaults to `True`.
            return_attention_mask (bool, optional):
                Whether to include the attention mask in the returned dictionary.
                Defaults to `False`.
            return_length (bool, optional):
                Whether to include the length of each encoded inputs in the
                returned dictionary. Defaults to `False`.
            return_overflowing_tokens (bool, optional):
                Whether to include overflowing token information in the returned
                dictionary. Defaults to `False`.
            return_special_tokens_mask (bool, optional):
                Whether to include special tokens mask information in the returned
                dictionary. Defaults to `False`.

        Returns:
            dict or list[dict] (for batch input):
                The dict has the following optional items:

                - **input_ids** (list[int]): List of token ids to be fed to a model.
                - **position_ids** (list[int], optional): List of token position ids to be
                  fed to a model. Included when `return_position_ids` is `True`
                - **token_type_ids** (list[int], optional): List of token type ids to be
                  fed to a model. Included when `return_token_type_ids` is `True`.
                - **attention_mask** (list[int], optional): List of integers valued 0 or 1,
                  where 0 specifies paddings and should not be attended to by the
                  model. Included when `return_attention_mask` is `True`.
                - **entity_ids** (list[int]): List of token ids to be fed to a model. Included when
                  `entity_spans` is not `None`.
                - **entity_position_ids** (list[int], optional): List of token position ids to be
                  fed to a model. Included when `entity_spans` is not `None`.
                - **entity_segment_ids** (list[int], optional): List of token type ids to be
                  fed to a model. Included when `entity_spans` is not `None`.
                - **entity_attention_mask** (list[int], optional): List of integers valued 0 or 1,
                  where 0 specifies paddings and should not be attended to by the
                  model. Included when `entity_spans` is not `None`.
                - **seq_len** (int, optional): The input_ids length. Included when `return_length`
                  is `True`.
                - **overflowing_tokens** (list[int], optional): List of overflowing tokens.
                  Included when if `max_length` is specified and `return_overflowing_tokens`
                  is True.
                - **num_truncated_tokens** (int, optional): The number of overflowing tokens.
                  Included when if `max_length` is specified and `return_overflowing_tokens`
                  is True.
                - **special_tokens_mask** (list[int], optional): List of integers valued 0 or 1,
                  with 0 specifying special added tokens and 1 specifying sequence tokens.
                  Included when `return_special_tokens_mask` is `True`.
                - **offset_mapping** (list[int], optional): list of pair preserving the
                  index of start and end char in original input for each token.
                  For a special token, the index pair is `(0, 0)`. Included when
                  `stride` works.
                - **overflow_to_sample** (int, optional): Index of example from which this
                  feature is generated. Included when `stride` works.
        """

        global _add_prefix_space
        if add_prefix_space:
            _add_prefix_space = True

        encode_output = super(LukeTokenizer, self).__call__(
            text,
            text_pair=text_pair,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            padding=padding,
            truncation=truncation,
            return_position_ids=return_position_ids,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_length=return_length,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            **kwargs,
        )
        if not entity_spans:
            return encode_output
        is_batched = bool(
            (not is_split_into_words and isinstance(text, (list, tuple)))
            or (
                is_split_into_words and isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
            )
        )
        if is_batched:
            if entities is None:
                entities = [None] * len(entity_spans)
            for i, ent in enumerate(zip(entities, entity_spans, text)):
                entity_encode = self.entity_encode(ent[2], ent[0], max_mention_length, ent[1])
                encode_output[i].update(entity_encode)
            if entity_spans_pair:
                if entities_pair is None:
                    entities_pair = [None] * len(entity_spans_pair)
                for i, ent in enumerate(zip(entities_pair, entity_spans_pair, text_pair)):
                    entity_encode = self.entity_encode(
                        ent[2],
                        ent[0],
                        max_mention_length,
                        ent[1],
                        1,
                        encode_output[i]["input_ids"].index(self.sep_token_id) + 2,
                    )
                    for k in entity_encode.keys():
                        encode_output[i][k] = encode_output[i][k] + entity_encode[k]

        else:
            entity_encode = self.entity_encode(text, entities, max_mention_length, entity_spans)

            encode_output.update(entity_encode)
            if entity_spans_pair:
                entity_encode = self.entity_encode(
                    text_pair,
                    entities_pair,
                    max_mention_length,
                    entity_spans_pair,
                    1,
                    encode_output["input_ids"].index(self.sep_token_id) + 2,
                )
                for k in entity_encode.keys():
                    encode_output[k] = encode_output[k] + entity_encode[k]

        return encode_output

    def tokenize(self, text, add_prefix_space=False):
        """
        Tokenize a string.
            Args:
              text (str):
                The sentence to be tokenized.
              add_prefix_space (boolean, default False):
                Begin the sentence with at least one space to get invariance
                to word order in GPT-2 (and Luke) tokenizers.
        """
        if _add_prefix_space:
            add_prefix_space = True

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.strip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text, add_prefix_space)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.added_tokens_encoder and sub_text not in self._all_special_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token, add_prefix_space)
                        if token not in self.added_tokens_encoder and token not in self._all_special_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        added_tokens = list(self.added_tokens_encoder.keys()) + self._all_special_tokens
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

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
        word = " ".join(word)
        self.cache[token] = word
        return word

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]

        return self._convert_token_to_id(token)

    def add_special_tokens(self, token_list: Union[List[int], Dict]):
        """
        Adding special tokens if you need.

        Args:
            token_list (List[int], Dict[List[int]]):
                The special token list you provided. If you provide a Dict, the key of the Dict must
                be "additional_special_tokens" and the value must be token list.
        """
        if isinstance(token_list, dict):
            token_list = token_list["additional_special_tokens"]
        encoder_dict = dict()
        decoder_dict = dict()
        for token in token_list:
            encoder_dict[token] = len(self.encoder.keys())
            decoder_dict[len(self.decoder.keys())] = token
        self.added_tokens_encoder.update(encoder_dict)
        self.added_tokens_decoder.update(decoder_dict)

    def convert_entity_to_id(self, entity: str):
        """Convert the entity to id"""
        if not self.entity_vocab.get(entity, None):
            warnings.warn(f"{entity} not found in entity thesaurus")
            return None
        else:
            return self.entity_vocab[entity]

    def entity_encode(self, text, entities, max_mention_length, entity_spans, ent_sep=0, offset_a=1):
        """Convert the string entity to digital entity"""

        def convert_tuple_to_list(x):
            """This function aim to convert tuple to list"""
            if isinstance(x, tuple):
                x = list(x)
            for i, each_x in enumerate(x):
                if isinstance(each_x, tuple):
                    x[i] = list(each_x)
            return x

        mentions = []
        if entities:
            for i, entity in enumerate(zip(entities, entity_spans)):
                entity = convert_tuple_to_list(entity)
                entity[1][0], entity[1][1] = self._convert_entity_pos(text, entity[1])
                if not self.entity_vocab.get(entity[0], None):
                    warnings.warn(f"{entity[0]} not found in entity thesaurus")
                    mentions.append((1, entity[1][0], entity[1][1]))
                else:
                    mentions.append((self.entity_vocab[entity[0]], entity[1][0], entity[1][1]))
        else:
            entities = [2] * len(entity_spans)
            for i, entity in enumerate(zip(entities, entity_spans)):
                entity = convert_tuple_to_list(entity)
                entity[1][0], entity[1][1] = self._convert_entity_pos(text, entity[1])
                mentions.append((entity[0], entity[1][0], entity[1][1]))

        entity_ids = [0] * len(mentions)
        entity_segment_ids = [ent_sep] * len(mentions)
        entity_attention_mask = [1] * len(mentions)
        entity_position_ids = [[-1 for y in range(max_mention_length)] for x in range(len(mentions))]

        for i, (offset, (entity_id, start, end)) in enumerate(zip(repeat(offset_a), mentions)):
            entity_ids[i] = entity_id
            entity_position_ids[i][: end - start] = range(start + offset, end + offset)
        return dict(
            entity_ids=entity_ids,
            entity_token_type_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids,
        )

    def _convert_entity_pos(self, text, entity_span):
        text_token = self.tokenize(text[0 : entity_span[0]].strip())
        entity_token = self.tokenize(text[entity_span[0] : entity_span[1]].strip())
        return len(text_token), len(text_token) + len(entity_token)

    def get_offset_mapping(self, text):
        tokens = self._tokenize(text)
        offset_mapping = []
        offset = 0
        for token in tokens:
            if token[0] == "Ġ":
                offset_mapping.append((offset + 1, offset + len(token)))
            else:
                offset_mapping.append((offset, offset + len(token)))
            offset += len(token)

        return offset_mapping

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.

        A Luke sequence pair mask has the following format:
        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (List[int]):
                A list of `inputs_ids` for the first sequence.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs. Defaults to None.

        Returns:
            List[int]: List of token_type_id according to the given sequence(s).
        """
        _sep = [self.sep_token_id]
        _cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(_cls + token_ids_0 + _sep) * [0]
        return len(_cls + token_ids_0 + _sep) * [0] + len(_sep + token_ids_1 + _sep) * [1]

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Args:
            pair(bool):
                Whether the input is a sequence pair or a single sequence.
                Defaults to `False` and the input is a single sequence.

        Returns:
            int: Number of tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

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
