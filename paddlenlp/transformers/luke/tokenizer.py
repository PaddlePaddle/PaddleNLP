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

from typing import Optional, Union, List, Dict

try:
    import regex as re
except:
    import re
import sys
import json
import itertools
from .. import RobertaTokenizer
from .entity_vocab import EntityVocab
from itertools import repeat
import warnings

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func


__all__ = ['LukeTokenizer']
_add_prefix_space = False


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


class LukeTokenizer(RobertaTokenizer):
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
            #{'input_ids': [0, 40401, 261, 12695, 1074, 11, 1287, 1422, 2],
            #'entity_ids': [1657, 32]

    """

    # resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "entity_file": "entity_vocab.tsv"
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "luke-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-base/vocab.json",
            "luke-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-large/vocab.json"
        },
        "merges_file": {
            "luke-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-base/merges.txt",
            "luke-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-large/merges.txt"
        },
        "entity_file": {
            "luke-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-base/entity_vocab.tsv",
            "luke-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/luke/luke-large/entity_vocab.tsv"
        },
    }
    pretrained_init_configuration = {
        "luke-base": {
            "do_lower_case": True
        },
        "luke-large": {
            "do_lower_case": True
        }
    }

    def __init__(self,
                 vocab_file,
                 entity_file,
                 merges_file,
                 do_lower_case=True,
                 unk_token="<unk>",
                 sep_token="</s>",
                 pad_token="<pad>",
                 cls_token="<s>",
                 mask_token="<mask>"):

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.entity_vocab = EntityVocab(entity_file)
        self.sep_token, self.sep_token_id = sep_token, self.encoder[sep_token]
        self.cls_token, self.cls_token_id = cls_token, self.encoder[cls_token]
        self.pad_token, self.pad_token_id = pad_token, self.encoder[pad_token]
        self.unk_token, self.unk_token_id = unk_token, self.encoder[unk_token]
        self._all_special_tokens = [
            unk_token, sep_token, pad_token, cls_token, mask_token
        ]
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = 'replace'  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding='utf-8') as merges_handle:
            bpe_merges = merges_handle.read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        super(LukeTokenizer, self).__init__(
            vocab_file,
            merges_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token)

    def get_entity_vocab(self):
        """Get the entity vocab"""
        return self.entity_vocab

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.decoder.get(index)

    def _tokenize(self, text, add_prefix_space=False):
        if add_prefix_space:
            text = ' ' + text

        bpe_tokens = []
        for token in re.findall(self.pat, text):
            if sys.version_info[0] == 2:
                token = ''.join(
                    self.byte_encoder[ord(b)] for b in token
                )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            else:
                token = ''.join(
                    self.byte_encoder[b] for b in token.encode('utf-8')
                )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
            bpe_tokens.extend(
                bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def __call__(self,
                 text,
                 text_pair=None,
                 entity_spans=None,
                 entity_spans_pair=None,
                 entities=None,
                 entities_pair=None,
                 max_mention_length=30,
                 max_seq_len: Optional[int]=None,
                 stride=0,
                 add_prefix_space=False,
                 is_split_into_words=False,
                 pad_to_max_seq_len=False,
                 truncation_strategy="longest_first",
                 return_position_ids=False,
                 return_token_type_ids=True,
                 return_attention_mask=True,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences, depending on the task you want to prepare them for.
        Args:
             text (`str`, `List[str]`, `List[List[str]]`):
                    The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
                    tokenizer does not support tokenization based on pretokenized strings.
             text_pair (`str`, `List[str]`, `List[List[str]]`):
                    The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
                    tokenizer does not support tokenization based on pretokenized strings.
             entity_spans (`List[Tuple[int, int]]`, `List[List[Tuple[int, int]]]`, *optional*):
                The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
                with two integers denoting character-based start and end positions of entities. If you specify
                `"entity_classification"` or `"entity_pair_classification"` as the `task` argument in the constructor,
                the length of each sequence must be 1 or 2, respectively. If you specify `entities`, the length of each
                sequence must be equal to the length of each sequence of `entities`.
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
        """
        global _add_prefix_space
        if add_prefix_space:
            _add_prefix_space = True

        encode_output = super(LukeTokenizer, self).__call__(
            text,
            text_pair=text_pair,
            max_seq_len=max_seq_len,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_max_seq_len=pad_to_max_seq_len,
            truncation_strategy=truncation_strategy,
            return_position_ids=return_position_ids,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_length=return_length,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask)
        if not entity_spans:
            return encode_output
        is_batched = bool(
            (not is_split_into_words and isinstance(text, (list, tuple))) or
            (is_split_into_words and isinstance(text, (list, tuple)) and
             text and isinstance(text[0], (list, tuple))))
        if is_batched:
            if entities is None:
                entities = [None] * len(entity_spans)
            for i, ent in enumerate(zip(entities, entity_spans)):
                entity_encode = self.entity_encode(ent[0], max_mention_length,
                                                   ent[1])
                encode_output[i].update(entity_encode)
            if entity_spans_pair:
                if entities_pair is None:
                    entities_pair = [None] * len(entity_spans_pair)
                for i, ent in enumerate(zip(entities_pair, entity_spans_pair)):
                    entity_encode = self.entity_encode(
                        ent[0], max_mention_length, ent[1], 1,
                        encode_output[i]['input_ids'].index(self.sep_token_id) +
                        1)
                    for k in entity_encode.keys():
                        encode_output[i][k] = encode_output[i][
                            k] + entity_encode[k]

        else:
            entity_encode = self.entity_encode(entities, max_mention_length,
                                               entity_spans)
            encode_output.update(entity_encode)
            if entity_spans_pair:
                entity_encode = self.entity_encode(
                    entities_pair, max_mention_length, entity_spans_pair, 1,
                    encode_output['input_ids'].index(self.sep_token_id) + 2)
            for k in entity_encode.keys():
                encode_output[k] = encode_output[k] + entity_encode[k]

        return encode_output

    def tokenize(self, text, add_prefix_space=False):
        """ Tokenize a string.
            Args:
              add_prefix_space (boolean, default False):
                Begin the sentence with at least one space to get invariance to word order in GPT-2 (and Luke) tokenizers.
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
                    if sub_text not in self.added_tokens_encoder \
                            and sub_text not in self._all_special_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return list(itertools.chain.from_iterable((self._tokenize(token, add_prefix_space) if token not \
                                                                                                  in self.added_tokens_encoder and token not in self._all_special_tokens \
                                                           else [token] for token in tokenized_text)))

        added_tokens = list(self.added_tokens_encoder.keys(
        )) + self._all_special_tokens
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

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors=self.errors)
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
            token_list = token_list['additional_special_tokens']
        encoder_dict = dict()
        decoder_dict = dict()
        for token in token_list:
            encoder_dict[token] = len(self.encoder.keys())
            decoder_dict[len(self.decoder.keys())] = token
        self.added_tokens_encoder.update(encoder_dict)
        self.added_tokens_decoder.update(decoder_dict)

    def convert_entity_to_id(self, entity: str):
        """Convert the entity to id"""
        if not self.entity_vocab[entity]:
            warnings.warn(f"{entity} not found in entity thesaurus")
            return None
        else:
            return self.entity_vocab[entity]

    def entity_encode(self,
                      entities,
                      max_mention_length,
                      entity_spans,
                      ent_sep=0,
                      offset_a=1):
        """Convert the string entity to digital entity"""
        mentions = []
        if entities:
            for i, entity in enumerate(zip(entities, entity_spans)):
                if not self.entity_vocab[entity[0]]:
                    warnings.warn(f"{entity[0]} not found in entity thesaurus")
                    mentions.append((1, entity[1][0], entity[1][1]))
                else:
                    mentions.append((self.entity_vocab[entity[0]], entity[1][0],
                                     entity[1][1]))
        else:
            entities = [2] * len(entity_spans)
            for i, entity in enumerate(zip(entities, entity_spans)):
                mentions.append((entity[0], entity[1][0], entity[1][1]))

        entity_ids = [0] * len(mentions)
        entity_segment_ids = [ent_sep] * len(mentions)
        entity_attention_mask = [1] * len(mentions)
        entity_position_ids = [[-1 for y in range(max_mention_length)]
                               for x in range(len(mentions))]

        for i, (offset, (entity_id, start,
                         end)) in enumerate(zip(repeat(offset_a), mentions)):
            entity_ids[i] = entity_id
            entity_position_ids[i][:end - start] = range(start + offset,
                                                         end + offset)

        return dict(
            entity_ids=entity_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
            entity_position_ids=entity_position_ids)
