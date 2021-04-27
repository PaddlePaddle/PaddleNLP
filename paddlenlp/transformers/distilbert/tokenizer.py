# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from ..bert.tokenizer import BertTokenizer

__all__ = ['DistilBertTokenizer']


class DistilBertTokenizer(BertTokenizer):
    """
    Constructs a DistilBERT tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "distilbert-base-uncased":
            "https://paddlenlp.bj.bcebos.com/models/distilbert/distilbert-base-uncased-vocab.txt",
            "distilbert-base-cased":
            "https://paddlenlp.bj.bcebos.com/models/distilbert/distilbert-base-cased-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "distilbert-base-uncased": {
            "do_lower_case": True
        },
        "distilbert-base-cased": {
            "do_lower_case": False
        },
    }

    def __call__(self,
                 text,
                 text_pair=None,
                 max_seq_len=None,
                 stride=0,
                 is_split_into_words=False,
                 pad_to_max_seq_len=False,
                 truncation_strategy="longest_first",
                 return_position_ids=False,
                 return_token_type_ids=False,
                 return_attention_mask=False,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        """
            Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
            sequences. This method will call `self.encode()` or `self.batch_encode()` depending on input format and  
            `is_split_into_words` argument.
            Args:
                text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                    The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                    (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                    :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                text_pair (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                    The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                    (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                    :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            """
        # Input type checking for clearer error
        assert isinstance(text, str) or (
            isinstance(text, (list, tuple)) and (len(text) == 0 or (
                isinstance(text[0], str) or
                (isinstance(text[0], (list, tuple)) and
                 (len(text[0]) == 0 or isinstance(text[0][0], str)))))
        ), ("text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
            "or `List[List[str]]` (batch of pretokenized examples).")

        assert (text_pair is None or isinstance(text_pair, str) or (
            isinstance(text_pair, (list, tuple)) and (len(text_pair) == 0 or (
                isinstance(text_pair[0], str) or
                (isinstance(text_pair[0], (list, tuple)) and
                 (len(text_pair[0]) == 0 or isinstance(text_pair[0][0], str)))))
        )), (
            "text_pair input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
            "or `List[List[str]]` (batch of pretokenized examples).")

        is_batched = bool(
            (not is_split_into_words and isinstance(text, (list, tuple))) or
            (is_split_into_words and isinstance(text, (list, tuple)) and
             text and isinstance(text[0], (list, tuple))))

        if is_batched:
            batch_text_or_text_pairs = list(zip(
                text, text_pair)) if text_pair is not None else text
            return self.batch_encode(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                max_seq_len=max_seq_len,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_max_seq_len=pad_to_max_seq_len,
                truncation_strategy="longest_first",
                return_position_ids=return_position_ids,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_length=return_length,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask)
        else:
            return self.encode(
                text=text,
                text_pair=text_pair,
                max_seq_len=max_seq_len,
                pad_to_max_seq_len=pad_to_max_seq_len,
                truncation_strategy="longest_first",
                return_position_ids=return_position_ids,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_length=return_length,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask)
