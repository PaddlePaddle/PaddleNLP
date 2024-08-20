# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from unittest.mock import MagicMock

import pytest

from paddlenlp.transformers import ChineseBertTokenizer


class TestChineseBertTokenizer:
    @pytest.fixture
    def tokenizer(self):
        # Load pretrained model and replace some methods
        tokenizer = ChineseBertTokenizer.from_pretrained("ChineseBERT-base")

        # Mock some methods
        tokenizer.tokenize = MagicMock(return_value=["[CLS]", "hello", "[SEP]"])
        tokenizer.convert_tokens_to_ids = MagicMock(side_effect=[[101, 7592, 102]])
        tokenizer.num_special_tokens_to_add = MagicMock(return_value=2)
        tokenizer.get_offset_mapping = MagicMock(return_value=[(0, 0), (0, 5), (0, 5)])
        tokenizer.truncate_sequences = MagicMock(return_value=([], [], [], [], []))
        tokenizer.build_inputs_with_special_tokens = MagicMock(return_value=[101, 7592, 102])
        tokenizer.create_token_type_ids_from_sequences = MagicMock(return_value=[0])
        tokenizer.get_pinyin_ids = MagicMock(return_value=[1, 2, 3])
        tokenizer.get_special_tokens_mask = MagicMock(return_value=[1, 1, 1])

        return tokenizer

    def test_encode_single_text(self, tokenizer):
        text = "hello"
        result = tokenizer.encode(text)

        tokenizer.tokenize.assert_called_once_with(text)
        tokenizer.convert_tokens_to_ids.assert_called_once_with(["[CLS]", "hello", "[SEP]"])
        assert result["input_ids"] == [101, 7592, 102]
        assert result["pinyin_ids"] == [1, 2, 3]
        assert "token_type_ids" in result
        assert "position_ids" not in result

    def test_encode_with_pair(self, tokenizer):
        text = "hello"
        text_pair = "world"
        tokenizer.tokenize.side_effect = [["[CLS]", "hello", "[SEP]"], ["[SEP]", "world", "[SEP]"]]

        result = tokenizer.encode(text, text_pair)

        assert result["input_ids"] == [
            101,
            7592,
            102,
            102,
            12345,
            102,
        ]  # Assuming convert_tokens_to_ids returns [12345] for "world"
        assert len(result["pinyin_ids"]) == 6

    def test_encode_with_max_seq_len(self, tokenizer):
        text = "a very long text that needs to be truncated"
        tokenizer.tokenize.return_value = ["[CLS]"] + ["word"] * 100 + ["[SEP]"]
        tokenizer.truncate_sequences.return_value = ([101] + [1234] * 50 + [102], None, [], [], [])

        result = tokenizer.encode(text, max_seq_len=52)

        assert len(result["input_ids"]) == 52
        assert result["input_ids"][:3] == [101, 1234, 1234]
        assert result["input_ids"][-1] == 102

    def test_encode_with_padding(self, tokenizer):
        text = "short"
        tokenizer.tokenize.return_value = ["[CLS]", "short", "[SEP]"]

        result = tokenizer.encode(text, max_seq_len=10, pad_to_max_seq_len=True)

        assert len(result["input_ids"]) == 10
        assert result["input_ids"][-1] == 0  # Assuming 0 is the pad token id
        assert result["pinyin_ids"][-7:] == [0] * 7  # Padding should be applied to pinyin_ids as well

    def test_encode_with_attention_mask(self, tokenizer):
        text = "hello"
        result = tokenizer.encode(text, return_attention_mask=True)

        assert result["attention_mask"] == [1, 1, 1]

    def test_encode_with_position_ids(self, tokenizer):
        text = "hello"
        result = tokenizer.encode(text, return_position_ids=True)

        assert result["position_ids"] == [0, 1, 2]

    def test_encode_with_overflowing_tokens(self, tokenizer):
        text = "a very long text"
        tokenizer.tokenize.return_value = ["[CLS]"] + ["word"] * 100 + ["[SEP]"]
        tokenizer.truncate_sequences.return_value = ([101] + [1234] * 49 + [102], None, [], ["word"] * 51, 51)

        result = tokenizer.encode(text, max_seq_len=50, return_overflowing_tokens=True)

        assert result["overflowing_tokens"] == ["word"] * 51
        assert result["num_truncated_tokens"] == 51
