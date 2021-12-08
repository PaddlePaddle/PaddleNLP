#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2021 ShannonAI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import lru_cache
from pypinyin import NORMAL, Style, pinyin

from paddlenlp.transformers import BertTokenizer


class ChineseBertTokenizer(BertTokenizer):
    """
    Construct a ChineseBert tokenizer. `ChineseBertTokenizer` is identical to `BertTokenizerr`.
    The difference between them is that ChineseBert has the extra process about pinyin id.
    For more information regarding those methods, please refer to this superclass.
    """
    pretrained_resource_files_map = {
        "vocab_file": {
            "ChineseBERT-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/chinese_bert/chinesebert-base/vocab.txt",
            "ChineseBERT-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/chinese_bert/chinesebert-base/tokenizer_config.json",
        },
        "tokenizer_config_file": {
            "ChineseBERT-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/chinese_bert/chinesebert-large/vocab.txt",
            "ChineseBERT-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/chinese_bert/chinesebert-large/tokenizer_config.json",
        },
    }
    pretrained_init_configuration = {
        "ChineseBERT-base": {
            "do_lower_case": True
        },
        "ChineseBERT-large": {
            "do_lower_case": True
        },
    }
    padding_side = "right"

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            pinyin_map=None,
            id2pinyin=None,
            pinyin2tensor=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]", ):
        super().__init__(
            vocab_file,
            do_lower_case,
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token, )
        self.pinyin_dict = pinyin_map
        self.id2pinyin = id2pinyin
        self.pinyin2tensor = pinyin2tensor
        self.special_tokens_pinyin_ids = [0] * 8

    def encode(self,
               text,
               text_pair=None,
               max_seq_len=512,
               pad_to_max_seq_len=False,
               truncation_strategy="longest_first",
               return_position_ids=False,
               return_token_type_ids=True,
               return_attention_mask=False,
               return_length=False,
               return_overflowing_tokens=False,
               return_special_tokens_mask=False):
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self._tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        ids = get_input_ids(text)
        pair_ids = get_input_ids(text_pair) if text_pair is not None else None

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}

        # Truncation: Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(
            pair=pair))

        token_offset_mapping = self.get_offset_mapping(text)

        if pair:
            token_pair_offset_mapping = self.get_offset_mapping(text_pair)
        else:
            token_pair_offset_mapping = None

        if max_seq_len and total_len > max_seq_len:
            ids, pair_ids, token_offset_mapping, token_pair_offset_mapping, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                token_offset_mapping=token_offset_mapping,
                token_pair_offset_mapping=token_pair_offset_mapping,
                num_tokens_to_remove=total_len - max_seq_len,
                truncation_strategy=truncation_strategy, )

            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_seq_len

        # Add special tokens

        sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
        token_type_ids = self.create_token_type_ids_from_sequences(ids,
                                                                   pair_ids)

        offset_mapping = self.build_offset_mapping_with_special_tokens(
            token_offset_mapping, token_pair_offset_mapping)

        # Build output dictionnary
        encoded_inputs["input_ids"] = sequence
        encoded_inputs["pinyin_ids"] = self.get_pinyin_ids(text, text_pair,
                                                           offset_mapping)

        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            encoded_inputs[
                "special_tokens_mask"] = self.get_special_tokens_mask(ids,
                                                                      pair_ids)
        if return_length:
            encoded_inputs["seq_len"] = len(encoded_inputs["input_ids"])

        # Check lengths
        assert max_seq_len is None or len(encoded_inputs[
            "input_ids"]) <= max_seq_len

        # Padding
        needs_to_be_padded = pad_to_max_seq_len and \
                             max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

        if needs_to_be_padded:
            difference = max_seq_len - len(encoded_inputs["input_ids"])
            encoded_inputs["pinyin_ids"] = encoded_inputs[
                "pinyin_ids"] + self.special_tokens_pinyin_ids * difference
            if self.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                        "input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] +
                        [self.pad_token_type_id] * difference)
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs[
                        "special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs[
                    "input_ids"] + [self.pad_token_id] * difference
            elif self.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [
                        1
                    ] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        [self.pad_token_type_id] * difference +
                        encoded_inputs["token_type_ids"])
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [
                    self.pad_token_id
                ] * difference + encoded_inputs["input_ids"]
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                    "input_ids"])

        if return_position_ids:
            encoded_inputs["position_ids"] = list(
                range(len(encoded_inputs["input_ids"])))

        return encoded_inputs

    def batch_encode(self,
                     batch_text_or_text_pairs,
                     max_seq_len=512,
                     pad_to_max_seq_len=False,
                     stride=0,
                     is_split_into_words=False,
                     truncation_strategy="longest_first",
                     return_position_ids=False,
                     return_token_type_ids=True,
                     return_attention_mask=False,
                     return_length=False,
                     return_overflowing_tokens=False,
                     return_special_tokens_mask=False):
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self._tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        batch_encode_inputs = []
        for example_id, tokens_or_pair_tokens in enumerate(
                batch_text_or_text_pairs):
            if not isinstance(tokens_or_pair_tokens, (list, tuple)):
                text, text_pair = tokens_or_pair_tokens, None
            elif is_split_into_words and not isinstance(
                    tokens_or_pair_tokens[0], (list, tuple)):
                text, text_pair = tokens_or_pair_tokens, None
            else:
                text, text_pair = tokens_or_pair_tokens

            if stride > 0 and text_pair is not None:
                first_ids = get_input_ids(text)
                second_ids = get_input_ids(text_pair)

                max_len_for_pair = max_seq_len - len(
                    first_ids) - self.num_special_tokens_to_add(pair=True)
                token_offset_mapping = self.get_offset_mapping(text)
                token_pair_offset_mapping = self.get_offset_mapping(text_pair)

                while True:
                    encoded_inputs = {}

                    ids = first_ids
                    mapping = token_offset_mapping
                    if len(second_ids) <= max_len_for_pair:
                        pair_ids = second_ids
                        pair_mapping = token_pair_offset_mapping
                    else:
                        pair_ids = second_ids[:max_len_for_pair]
                        pair_mapping = token_pair_offset_mapping[:
                                                                 max_len_for_pair]

                    offset_mapping = self.build_offset_mapping_with_special_tokens(
                        mapping, pair_mapping)
                    # add_pinyin_ids
                    encoded_inputs["pinyin_ids"] = self.get_pinyin_ids(
                        text, text_pair, offset_mapping)

                    sequence = self.build_inputs_with_special_tokens(ids,
                                                                     pair_ids)
                    token_type_ids = self.create_token_type_ids_from_sequences(
                        ids, pair_ids)

                    # Build output dictionnary
                    encoded_inputs["input_ids"] = sequence
                    if return_token_type_ids:
                        encoded_inputs["token_type_ids"] = token_type_ids
                    if return_special_tokens_mask:
                        encoded_inputs[
                            "special_tokens_mask"] = self.get_special_tokens_mask(
                                ids, pair_ids)
                    if return_length:
                        encoded_inputs["seq_len"] = len(encoded_inputs[
                            "input_ids"])

                    # Check lengths
                    assert max_seq_len is None or len(encoded_inputs[
                        "input_ids"]) <= max_seq_len

                    # Padding
                    needs_to_be_padded = pad_to_max_seq_len and \
                                        max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

                    encoded_inputs['offset_mapping'] = offset_mapping

                    if needs_to_be_padded:
                        difference = max_seq_len - len(encoded_inputs[
                            "input_ids"])
                        # padding pinyin_ids
                        encoded_inputs["pinyin_ids"] = encoded_inputs[
                            "pinyin_ids"] + self.special_tokens_pinyin_ids * difference
                        if self.padding_side == 'right':
                            if return_attention_mask:
                                encoded_inputs["attention_mask"] = [1] * len(
                                    encoded_inputs[
                                        "input_ids"]) + [0] * difference
                            if return_token_type_ids:
                                # 0 for padding token mask
                                encoded_inputs["token_type_ids"] = (
                                    encoded_inputs["token_type_ids"] +
                                    [self.pad_token_type_id] * difference)
                            if return_special_tokens_mask:
                                encoded_inputs[
                                    "special_tokens_mask"] = encoded_inputs[
                                        "special_tokens_mask"] + [1
                                                                  ] * difference
                            encoded_inputs["input_ids"] = encoded_inputs[
                                "input_ids"] + [self.pad_token_id] * difference
                            encoded_inputs['offset_mapping'] = encoded_inputs[
                                'offset_mapping'] + [(0, 0)] * difference
                        elif self.padding_side == 'left':
                            if return_attention_mask:
                                encoded_inputs["attention_mask"] = [
                                    0
                                ] * difference + [1] * len(encoded_inputs[
                                    "input_ids"])
                            if return_token_type_ids:
                                # 0 for padding token mask
                                encoded_inputs["token_type_ids"] = (
                                    [self.pad_token_type_id] * difference +
                                    encoded_inputs["token_type_ids"])
                            if return_special_tokens_mask:
                                encoded_inputs["special_tokens_mask"] = [
                                    1
                                ] * difference + encoded_inputs[
                                    "special_tokens_mask"]
                            encoded_inputs["input_ids"] = [
                                self.pad_token_id
                            ] * difference + encoded_inputs["input_ids"]
                            encoded_inputs['offset_mapping'] = [
                                (0, 0)
                            ] * difference + encoded_inputs['offset_mapping']
                    else:
                        if return_attention_mask:
                            encoded_inputs["attention_mask"] = [1] * len(
                                encoded_inputs["input_ids"])

                    if return_position_ids:
                        encoded_inputs["position_ids"] = list(
                            range(len(encoded_inputs["input_ids"])))

                    encoded_inputs['overflow_to_sample'] = example_id
                    batch_encode_inputs.append(encoded_inputs)

                    if len(second_ids) <= max_len_for_pair:
                        break
                    else:
                        second_ids = second_ids[max_len_for_pair - stride:]
                        token_pair_offset_mapping = token_pair_offset_mapping[
                            max_len_for_pair - stride:]

            else:
                batch_encode_inputs.append(
                    self.encode(
                        text,
                        text_pair,
                        max_seq_len=max_seq_len,
                        pad_to_max_seq_len=pad_to_max_seq_len,
                        truncation_strategy=truncation_strategy,
                        return_position_ids=return_position_ids,
                        return_token_type_ids=return_token_type_ids,
                        return_attention_mask=return_attention_mask,
                        return_length=return_length,
                        return_overflowing_tokens=return_overflowing_tokens,
                        return_special_tokens_mask=return_special_tokens_mask))

        return batch_encode_inputs

    def truncate_sequences(self,
                           ids,
                           pair_ids=None,
                           token_offset_mapping=None,
                           token_pair_offset_mapping=None,
                           num_tokens_to_remove=0,
                           truncation_strategy='longest_first',
                           stride=0):
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if truncation_strategy == 'longest_first':
            overflowing_tokens = []
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    overflowing_tokens = [ids[-1]] + overflowing_tokens
                    ids = ids[:-1]
                    token_offset_mapping = token_offset_mapping[:-1]
                else:
                    pair_ids = pair_ids[:-1]
                    token_pair_offset_mapping = token_pair_offset_mapping[:-1]
            window_len = min(len(ids), stride)
            if window_len > 0:
                overflowing_tokens = ids[-window_len:] + overflowing_tokens
        elif truncation_strategy == 'only_first':
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
            token_offset_mapping = token_offset_mapping[:-num_tokens_to_remove]
        elif truncation_strategy == 'only_second':
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
            token_pair_offset_mapping = token_pair_offset_mapping[:
                                                                  -num_tokens_to_remove]
        elif truncation_strategy == 'do_not_truncate':
            raise ValueError(
                "Input sequence are too long for max_length. Please select a truncation strategy."
            )
        else:
            raise ValueError(
                "Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', 'do_not_truncate']"
            )
        return (ids, pair_ids, token_offset_mapping, token_pair_offset_mapping,
                overflowing_tokens)

    @lru_cache(9999)
    def pinyin_locs_map(self, text):
        pinyin_list = pinyin(
            text,
            style=Style.TONE3,
            heteronym=True,
            errors=lambda x: [["not chinese"] for _ in x], )
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids
        return pinyin_locs

    def get_pinyin_ids(self, text, text_pair=None, offset_mapping=None):

        text_pinyin_locs = self.pinyin_locs_map(text)
        if text_pair:
            text_pair_pinyin_locs = self.pinyin_locs_map(text_pair)
        else:
            text_pair_pinyin_locs = None

        pinyin_ids = []
        special_token_count = 0

        for offset in offset_mapping:
            if offset == (0, 0):
                special_token_count += 1

            if special_token_count <= 1:
                pinyin_locs_maps = text_pinyin_locs
            else:
                pinyin_locs_maps = text_pair_pinyin_locs

            if offset[1] - offset[0] != 1:
                pinyin_ids.extend([0] * 8)
                continue
            if offset[0] in pinyin_locs_maps:
                pinyin_ids.extend(pinyin_locs_maps[offset[0]])
            else:
                pinyin_ids.extend([0] * 8)

        return pinyin_ids
