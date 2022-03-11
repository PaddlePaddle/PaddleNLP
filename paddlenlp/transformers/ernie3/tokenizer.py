# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.

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
import pickle
import re
import six
import shutil

from paddle.utils import try_import
from paddlenlp.utils.env import MODEL_HOME

from .. import BasicTokenizer, PretrainedTokenizer, WordpieceTokenizer

__all__ = ['Ernie3Tokenizer']


class Ernie3Tokenizer(PretrainedTokenizer):
    r"""
    Constructs an ERNIE tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        vocab_file (str): 
            The vocabulary file path (ends with '.txt') required to instantiate
            a `WordpieceTokenizer`.
        do_lower_case (str, optional): 
            Whether or not to lowercase the input when tokenizing.
            Defaults to`True`.
        unk_token (str, optional): 
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "[UNK]".
        sep_token (str, optional): 
            A special token separating two different sentences in the same input.
            Defaults to "[SEP]".
        pad_token (str, optional): 
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "[PAD]".
        cls_token (str, optional): 
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to "[CLS]".
        mask_token (str, optional): 
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to "[MASK]".
        s_token (str, optional): 
            A special token for splitting Chinese. Defaults to "<S>".
        end_token (str, optional): 
            A special token representing a end token.
            Defaults to "[END]".
    
    Examples:
        .. code-block::

            from paddlenlp.transformers import Ernie3Tokenizer
            tokenizer = Ernie3Tokenizer.from_pretrained('ernie3-10b')

            encoded_inputs = tokenizer('He was a puppeteer')
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie3-10b":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie3/vocab.txt"
        }
    }
    pretrained_init_configuration = {"ernie3-10b": {"do_lower_case": True}}

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 s_token="<S>",
                 end_token="[END]",
                 **kwargs):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file))
        self.do_lower_case = do_lower_case
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=unk_token)

    def __call__(self,
                 text,
                 text_pair=None,
                 max_seq_len=None,
                 stride=0,
                 is_split_into_words=False,
                 pad_to_max_seq_len=False,
                 truncation_strategy="longest_first",
                 return_position_ids=True,
                 return_token_type_ids=False,
                 return_attention_mask=False,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        return super(Ernie3Tokenizer, self).__call__(
            text, text_pair, max_seq_len, stride, is_split_into_words,
            pad_to_max_seq_len, truncation_strategy, return_position_ids,
            return_token_type_ids, return_attention_mask, return_length,
            return_overflowing_tokens, return_special_tokens_mask)

    @property
    def vocab_size(self):
        """
        Return the size of vocabulary.

        Returns:
            int: The size of vocabulary.
        """
        return len(self.vocab)

    def cut_sent(self, para):
        """ cut sent """
        # 单字符断句符
        para = re.sub('([。！？\?\!；;])([^。！？\?\!；;”’])', r"\1\n\2", para)

        # 双字符断句，双引号前有终止符，那么双引号才是句子的终点，
        para = re.sub('([。！？\?\!；;][。！？\?\!；;”’])([^。！？\?\!；;”’])', r'\1\n\2',
                      para)

        # 三字符
        para = re.sub('(？！”)([^”’])', r"\1\n\2", para)
        para = re.sub('([！。]’”)([^”’])', r"\1\n\2", para)
        para = re.sub('(\…{2}”)([^”’])', r"\1\n\2", para)

        # 英文省略号
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
        # 中文省略号  
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)

        #把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        #段尾如果有多余的\n就去掉它
        para = para.rstrip()
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，
        # 需要的再做些简单调整即可。
        sents = para.split("\n")
        sents = [sent for sent in sents if len(sent) != 0]
        return sents

    def is_chinese(self, word):
        """ is_chinese """
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    def _tokenize(self, text):
        r"""
        End-to-end tokenization for ERNIE models.

        Args:
            text (str): The text to be tokenized.
        
        Returns:
            List[str]: A list of string representing converted tokens.
        """
        split_tokens = []
        text_ls = text.split()  # 分隔后判断是否增加<S>
        for text_i, text in enumerate(text_ls):
            for sent in self.cut_sent(text):
                for token in self.basic_tokenizer.tokenize(sent):
                    for sub_token in self.wordpiece_tokenizer.tokenize(token):
                        split_tokens.append(sub_token)
                split_tokens.append(self.s_token)
            if text_i + 1 < len(text_ls):
                # 只有中文之间才加<S>
                if len(text) and self.is_chinese(text[-1]) and len(text_ls[
                        text_i + 1]) and self.is_chinese(text_ls[text_i + 1][
                            0]):
                    pass
                else:
                    split_tokens = split_tokens[:-1]
        if len(split_tokens) > 1 and split_tokens[
                -2] not in '。！？?!；;”’' and split_tokens[-1] == '<S>':
            split_tokens = split_tokens[:-1]
        if len(split_tokens) == 1 and split_tokens[0] == '<S>':
            split_tokens = []
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        r"""
        Converts a sequence of tokens (list of string) in a single string. Since
        the usage of WordPiece introducing `##` to concat subwords, also remove
        `##` when converting.

        Args:
            tokens (List[str]): A list of string representing tokens to be converted.

        Returns:
            str: Converted string from tokens.

        Examples:
            .. code-block::

                from paddlenlp.transformers import Ernie3Tokenizer
                tokenizer = Ernie3Tokenizer.from_pretrained('ernie3-10b')

                tokens = tokenizer.tokenize('He was a puppeteer')
                strings = tokenizer.convert_tokens_to_string(tokens)
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def convert_ids_to_string(self, ids):
        """
        Converts a sequence of tokens (strings for sub-words) in a single string.
        """
        tokens = self.convert_ids_to_tokens(ids)
        out_string = self.convert_tokens_to_string(tokens)
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        r"""
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. 
            Do not put this inside your training loop.

        Args:
            pair (bool, optional):
                Whether the input is a sequence pair or a single sequence.
                Defaults to `False` and the input is a single sequence.

        Returns:
            int: Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        r"""
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        An Ernie3 sequence has the following format:

        - single sequence:      ``[CLS] X [SEP]``
        - pair of sequences:        ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.
                Defaults to `None`.

        Returns:
            List[int]: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0
        _cls = [self.cls_token_id]
        return _cls + token_ids_0 + token_ids_1

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        r"""
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens. 
        
        An ERNIE3 offset_mapping has the following format:

        - single sequence:      ``(0,0) X (0,0)``
        - pair of sequences:        ``(0,0) A (0,0) B (0,0)``
        
        Args:
            offset_mapping_ids_0 (List[tuple]):
                List of char offsets to which the special tokens will be added.
            offset_mapping_ids_1 (List[tuple], optional):
                Optional second list of wordpiece offsets for offset mapping pairs.
                Defaults to `None`.

        Returns:
            List[tuple]: A list of wordpiece offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0

        return [(0, 0)] + offset_mapping_0 + offset_mapping_1
