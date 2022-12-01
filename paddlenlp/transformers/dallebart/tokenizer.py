# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021-2022 The Fairseq Authors and The Google Flax
# Team Authors And The HuggingFace Inc. team and & DALL·E Mini team.
# All rights reserved.
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

import html
import math
import random
import re
from pathlib import Path

from paddle.utils import try_import
from ...transformers import GPTTokenizer, AddedToken

__all__ = ["DalleBartTokenizer"]

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "dalle-mini": 64,
    "dalle-mega-v16": 64,
    "dalle-mega-v26": 64,
    "dalle-mega": 64,
}

# based on wiki word occurrence
person_token = [("a person", 282265), ("someone", 121194), ("somebody", 12219)]
temp_token = "xtokx"  # avoid repeating chars


class HashtagProcessor:
    # Adapted from wordninja library
    # We use our wikipedia word count + a good heuristic to make it work
    def __init__(self, wiki_word_frequency):
        self._word_cost = (l.split()[0] for l in Path(wiki_word_frequency).read_text(encoding="utf8").splitlines())
        self._word_cost = {str(k): math.log(float(i + 1)) for i, k in enumerate(self._word_cost)}
        self._max_word = max(len(x) for x in self._word_cost.keys())
        self._SPLIT_RE = re.compile("[^a-zA-Z0-9']+")

    def __call__(self, s):
        """Uses dynamic programming to infer the location of spaces in a string without spaces."""
        l = [self._split(x) for x in self._SPLIT_RE.split(s)]
        return " ".join([item for sublist in l for item in sublist])

    def _split(self, s):
        # Find the best match for the i first characters, assuming cost has
        # been built for the i-1 first characters.
        # Returns a pair (match_cost, match_length).
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i - self._max_word) : i]))
            return min((c + self._word_cost.get(s[i - k - 1 : i].lower(), 9e999), k + 1) for k, c in candidates)

        # Build the cost array
        cost = [0]
        for i in range(1, len(s) + 1):
            c, k = best_match(i)
            cost.append(c)

        # Backtrack to recover the minimal-cost string.
        out = []
        i = len(s)
        while i > 0:
            c, k = best_match(i)
            assert c == cost[i]
            newToken = True
            if not s[i - k : i] == "'":  # ignore a lone apostrophe
                if len(out) > 0:
                    # re-attach split 's and split digits
                    if out[-1] == "'s" or (s[i - 1].isdigit() and out[-1][0].isdigit()):  # digit followed by digit
                        out[-1] = s[i - k : i] + out[-1]  # combine current token with previous token
                        newToken = False

            if newToken:
                out.append(s[i - k : i])

            i -= k

        return reversed(out)


def replace_person_token(t):
    "Used for CC12M"
    t = re.sub(r"<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t)
    while "<person>" in t:
        t = t.replace("<person>", f" {random.choices(*tuple(zip(*person_token)))[0]} ", 1)
    return t


def fix_html(t):
    # from OpenAI CLIP
    return html.unescape(html.unescape(t))


def replace_punctuation_with_commas(t):
    return re.sub(r"[()[\].,|:;?!=+~\-\/{}]", ",", t)


def simplify_quotes(t):
    return re.sub("""['"`]""", ' " ', t)


def merge_quotes(t):
    return re.sub(r'(\s*"+\s*)+', ' " ', t)


def remove_comma_numbers(t):
    def _f(t):
        return re.sub(r"(\d),(\d{3})", r"\1\2", t)

    return _f(_f(t))


def pre_process_dot_numbers(t):
    return re.sub(r"(\w)\.(\w)", rf"\1{temp_token}dot{temp_token}\2", t)


def post_process_dot_numbers(t):
    return re.sub(f"{temp_token}dot{temp_token}", ".", t)


def pre_process_quotes(t):
    # allows quotes only for 's, 't, 'd, 'm, 'll, 're, 've
    return re.sub(r"'(?=([stdm]|(ll)|(re)|(ve)|(ll))\b)", rf"{temp_token}quote{temp_token}", t)


def post_process_quotes(t):
    return re.sub(f"{temp_token}quote{temp_token}", "'", t)


def pre_process_dates(t):
    return re.sub(r"(\d)/(\d)", rf"\1{temp_token}slash{temp_token}\2", t)


def post_process_dates(t):
    return re.sub(f"{temp_token}slash{temp_token}", "/", t)


def merge_commas(t):
    return re.sub(r"(\s*,+\s*)+", ", ", t)


def add_space_after_commas(t):
    return re.sub(",", ", ", t)


def handle_special_chars(t):
    "Handle special characters"
    # replace "-" with a space when between words without space
    t = re.sub(r"(\w)-(\w)", r"\1 \2", t)
    # always add space around some characters
    return re.sub(r"([%&\/$*])", r" \1 ", t)


def expand_hashtags(t, hashtag_processor):
    "Remove # and try to split words"
    return re.sub(r"#(\w+)", lambda m: hashtag_processor(m.group(1)), t)


_re_ignore_chars = r"[_#\\]"


def ignore_chars(t):
    "Ignore useless characters"
    return re.sub(_re_ignore_chars, " ", t)


def remove_extra_spaces(t):
    "Remove extra spaces (including \t and \n)"
    return re.sub(r"\s+", " ", t)


def remove_repeating_chars(t):
    "If the same character is present 4+ times (not 3 because of roman 'VIII'), replace with single instance"
    return re.sub(r"(\D)(\1{3,})", r"\1", t)


def remove_urls(t):
    return re.sub(r"http\S+", "", t)


def remove_html_tags(t):
    return re.sub("<[^<]+?>", " ", t)


def remove_first_last_commas(t):
    t = t.strip()
    t = t[:-1] if t and t[-1] == "," else t
    t = t[1:] if t and t[0] == "," else t
    return t.strip()


def remove_wiki_ref(t):
    t = re.sub(r"\A\s*\[\d+\]", "", t)
    return re.sub(r"\[\d+\]\s*\Z", "", t)


class TextNormalizer:
    def __init__(self, wiki_word_frequency_file):
        self._hashtag_processor = HashtagProcessor(wiki_word_frequency_file)
        self.emoji = try_import("emoji")
        self.ftfy = try_import("ftfy")
        self.unidecode = try_import("unidecode")

    def __call__(self, t):
        # fix some characters
        t = self.ftfy.fix_text(t)
        # fix html
        t = fix_html(t)
        # decode emojis (would be removed by unidecode)
        t = self.emoji.demojize(t)
        # decode and simplify text: see unidecode library
        t = self.unidecode.unidecode(t)
        # lower case
        t = t.lower()
        # replace <PERSON> (for CC12M)
        t = replace_person_token(t)
        # remove wiki reference (for WIT)
        t = remove_wiki_ref(t)
        # remove html tags
        t = remove_html_tags(t)
        # remove urls
        t = remove_urls(t)
        # remove commas in numbers
        t = remove_comma_numbers(t)
        # handle dots in numbers and quotes - Part 1
        t = pre_process_dot_numbers(t)
        t = pre_process_quotes(t)
        t = pre_process_dates(t)
        # handle special characters
        t = handle_special_chars(t)
        # handle hashtags
        t = expand_hashtags(t, self._hashtag_processor)
        # ignore useless characters
        t = ignore_chars(t)
        # simplify quotes
        t = simplify_quotes(t)
        # all punctuation becomes commas
        t = replace_punctuation_with_commas(t)
        # handle dots in numbers and quotes - Part 2
        t = post_process_dot_numbers(t)
        t = post_process_quotes(t)
        t = post_process_dates(t)
        # handle repeating characters
        t = remove_repeating_chars(t)
        # merge quotes
        t = merge_quotes(t)
        # merge commas
        t = merge_commas(t)
        # remove multiple spaces
        t = remove_extra_spaces(t)
        # remove first and last comma
        t = remove_first_last_commas(t)
        # always start with a space
        return f" {t}"


class DalleBartTokenizer(GPTTokenizer):
    r"""
    Construct a DalleBart tokenizer based on byte-level Byte-Pair-Encoding.

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
        wiki_word_frequency_file (str):
            Path to the wiki_word_frequency file when we need normlize text.
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

            from paddlenlp.transformers import DalleBartTokenizer

            tokenizer = DalleBartTokenizer.from_pretrained('dalle-mini')
            print(tokenizer('Donald Trump in Animal Crossing'))

            # {'input_ids': [0, 7083, 3252, 91, 2203, 7807, 2]}

    """
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "wiki_word_frequency_file": "enwiki-words-frequency.txt",
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "dalle-mini": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mini/vocab.json",
            "dalle-mega-v16": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v16/vocab.json",
            "dalle-mega-v26": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v26/vocab.json",
            "dalle-mega": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v26/vocab.json",
        },
        "merges_file": {
            "dalle-mini": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mini/merges.txt",
            "dalle-mega-v16": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v16/merges.txt",
            "dalle-mega-v26": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v26/merges.txt",
            "dalle-mega": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v26/merges.txt",
        },
        "wiki_word_frequency_file": {
            "dalle-mini": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mini/enwiki-words-frequency.txt",
            "dalle-mega-v16": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v16/enwiki-words-frequency.txt",
            "dalle-mega-v26": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v26/enwiki-words-frequency.txt",
            "dalle-mega": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v26/enwiki-words-frequency.txt",
        },
    }
    pretrained_init_configuration = {
        "dalle-mini": {"normalize_text": True},
        "dalle-mega-v16": {"normalize_text": True},
        "dalle-mega-v26": {"normalize_text": True},
        "dalle-mega": {"normalize_text": True},
    }
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        merges_file,
        wiki_word_frequency_file,
        normalize_text=True,
        errors="replace",
        max_len=None,
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
        self.normalize_text = normalize_text
        # in order to save wiki_word_frequency_file, we need set this attr
        self._wiki_word_frequency_file = wiki_word_frequency_file
        if self.normalize_text:
            self.text_processor = TextNormalizer(wiki_word_frequency_file)
        super().__init__(vocab_file, merges_file, errors, max_len, pad_token, eos_token, unk_token)

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

    def __call__(
        self,
        text,
        text_pair=None,
        max_length=64,  # default
        stride=0,
        is_split_into_words=False,
        padding="max_length",  # default
        truncation=True,  # default
        return_position_ids=False,
        return_token_type_ids=False,  # don't return token_type_ids
        return_attention_mask=True,  # default
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
        if self.normalize_text:
            is_batched = isinstance(text, (list, tuple))
            if is_batched:
                text = [self.text_processor(t) for t in text]
            else:
                text = self.text_processor(text)
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
