# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os

from .. import BasicTokenizer, PretrainedTokenizer, WordpieceTokenizer

__all__ = ['ElectraTokenizer', ]


class ElectraTokenizer(PretrainedTokenizer):
    """
    Constructs a Electra tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.
    Args:
        vocab_file (str): file path of the vocabulary
        do_lower_case (bool): Whether the text strips accents and convert to
            lower case. Default: `True`.
            Default: True.
        unk_token (str): The special token for unkown words. Default: "[UNK]".
        sep_token (str): The special token for separator token . Default: "[SEP]".
        pad_token (str): The special token for padding. Default: "[PAD]".
        cls_token (str): The special token for cls. Default: "[CLS]".
        mask_token (str): The special token for mask. Default: "[MASK]".
    
    Examples:
        .. code-block:: python
            from paddlenlp.transformers import ElectraTokenizer
            tokenizer = ElectraTokenizer.from_pretrained('electra-small-discriminator')
            # the following line get: ['he', 'was', 'a', 'puppet', '##eer']
            tokens = tokenizer('He was a puppeteer')
            # the following line get: 'he was a puppeteer'
            tokenizer.convert_tokens_to_string(tokens)
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "electra-small":
            "https://paddlenlp.bj.bcebos.com/models/transformers/electra/electra-small-vocab.txt",
            "electra-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/electra/electra-base-vocab.txt",
            "electra-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/electra/electra-large-vocab.txt",
            "chinese-electra-base":
            "http://paddlenlp.bj.bcebos.com/models/transformers/chinese-electra-base/vocab.txt",
            "chinese-electra-small":
            "http://paddlenlp.bj.bcebos.com/models/transformers/chinese-electra-small/vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "electra-small": {
            "do_lower_case": True
        },
        "electra-base": {
            "do_lower_case": True
        },
        "electra-large": {
            "do_lower_case": True
        },
        "chinese-electra-base": {
            "do_lower_case": True
        },
        "chinese-electra-small": {
            "do_lower_case": True
        }
    }

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]"):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = ElectraTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=unk_token)

    @property
    def vocab_size(self):
        """
        return the size of vocabulary.
        Returns:
            int: the size of vocabulary.
        """
        return len(self.vocab)

    def _tokenize(self, text):
        """
        End-to-end tokenization for Electra models.
        Args:
            text (str): The text to be tokenized.
        
        Returns:
            list: A list of string representing converted tokens.
        """
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def __call__(self, text):
        """
        End-to-end tokenization for Electra models.
        Args:
            text (str): The text to be tokenized.
        
        Returns:
            list: A list of string representing converted tokens.
        """
        return self._tokenize(text)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (list of string) in a single string. Since
        the usage of WordPiece introducing `##` to concat subwords, also remove
        `##` when converting.
        Args:
            tokens (list): A list of string representing tokens to be converted.
        Returns:
            str: Converted string from tokens.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.
        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.
        Args:
            pair: Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.
        Returns:
            Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. 
        
        A BERT sequence has the following format:
        ::
            - single sequence: ``[CLS] X [SEP]``
            - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. 
        A BERT sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of token_type_id according to the given sequence(s).
        """
        _sep = [self.sep_token_id]
        _cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(_cls + token_ids_0 + _sep) * [0]
        return len(_cls + token_ids_0 + _sep) * [0] + len(token_ids_1 +
                                                          _sep) * [1]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.

        Args:
            token_ids_0 (List[int]): List of ids of the first sequence.
            token_ids_1 (List[int], optinal): List of ids of the second sequence.
            already_has_special_tokens (bool, optional): Whether or not the token list is already 
                formatted with special tokens for the model. Defaults to None.

        Returns:
            results (List[int]): The list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def encode(self,
               text,
               text_pair=None,
               max_seq_len=None,
               pad_to_max_seq_len=True,
               truncation_strategy="longest_first",
               return_position_ids=True,
               return_segment_ids=True,
               return_input_mask=True,
               return_length=True,
               return_overflowing_tokens=False,
               return_special_tokens_mask=False):
        """
        Returns a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_seq_len`` is specified.
        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            max_seq_len (:obj:`int`, `optional`, defaults to :int:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            pad_to_max_seq_len (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the
                model's max length.
            truncation_strategy (:obj:`str`, `optional`, defaults to `longest_first`):
                String selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_seq_len
                  starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_seq_len)
            return_position_ids (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Set to True to return tokens position ids (default True).
            return_segment_ids (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to return token type IDs.
            return_input_mask (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to return the attention mask.
            return_length (:obj:`int`, defaults to :obj:`True`):
                If set the resulting dictionary will include the length of each encoded inputs
            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return overflowing token information (default False).
            return_special_tokens_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return special tokens mask information (default False).
        Return:
            A Dictionary of shape::
                {
                    input_ids: list[int],
                    position_ids: list[int] if return_position_ids is True (default)
                    segment_ids: list[int] if return_segment_ids is True (default)
                    input_mask: list[int] if return_input_mask is True (default)
                    seq_len: int if return_length is True (default)
                    overflowing_tokens: list[int] if a ``max_seq_len`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_seq_len`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if return_special_tokens_mask is True
                }
            With the fields:
            - ``input_ids``: list of token ids to be fed to a model
            - ``position_ids``: list of token position ids to be fed to a model
            - ``segment_ids``: list of token type ids to be fed to a model
            - ``input_mask``: list of indices specifying which tokens should be attended to by the model
            - ``length``: the input_ids length
            - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
            - ``num_truncated_tokens``: number of overflowing tokens a ``max_seq_len`` is specified
            - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
              tokens and 1 specifying sequence tokens.
        """

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
        if max_seq_len and total_len > max_seq_len:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_seq_len,
                truncation_strategy=truncation_strategy, )
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_seq_len

        # Add special tokens
        sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
        segment_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)

        # Build output dictionnary
        encoded_inputs["input_ids"] = sequence
        if return_segment_ids:
            encoded_inputs["segment_ids"] = segment_ids
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
            if return_input_mask:
                encoded_inputs["input_mask"] = [1] * len(encoded_inputs[
                    "input_ids"]) + [0] * difference
            if return_segment_ids:
                # 0 for padding token mask
                encoded_inputs["segment_ids"] = (
                    encoded_inputs["segment_ids"] + [0] * difference)
            if return_special_tokens_mask:
                encoded_inputs["special_tokens_mask"] = encoded_inputs[
                    "special_tokens_mask"] + [1] * difference
            encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [
                self.pad_token_id
            ] * difference
        else:
            if return_input_mask:
                encoded_inputs["input_mask"] = [1] * len(encoded_inputs[
                    "input_ids"])

        if return_position_ids:
            encoded_inputs["position_ids"] = list(
                range(len(encoded_inputs["input_ids"])))

        return encoded_inputs
