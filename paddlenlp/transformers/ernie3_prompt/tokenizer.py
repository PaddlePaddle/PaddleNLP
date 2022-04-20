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
import numpy as np
import paddle
from paddle.utils import try_import
from paddlenlp.utils.env import MODEL_HOME

from .. import BasicTokenizer, PretrainedTokenizer, WordpieceTokenizer

__all__ = ['Ernie3PromptTokenizer']


class Ernie3PromptTokenizer(PretrainedTokenizer):
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
            "ernie3-prompt":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie3/vocab_ernie3_prompt.txt"
        }
    }
    pretrained_init_configuration = {"ernie3-prompt": {"do_lower_case": False}}

    def __init__(self,
                 vocab_file,
                 do_lower_case=False,
                 add_s=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 s_token="[<S>]",
                 start_token="[START]",
                 gmask_token="[gMASK]",
                 gend_token="[gEND]",
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
        self.add_s = add_s

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
        return super(Ernie3PromptTokenizer, self).__call__(
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

    def _tokenize(self, text):
        r"""
        End-to-end tokenization for ERNIE models.

        Args:
            text (str): The text to be tokenized.
        
        Returns:
            List[str]: A list of string representing converted tokens.
        """
        split_tokens = []
        for sent in self.cut_sent(text):
            for token in self.basic_tokenizer.tokenize(sent):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
            if self.add_s:
                split_tokens.append(self.s_token)
        if len(split_tokens) >= 2 and split_tokens[
                -2] not in '。！？?!；;”"’' and split_tokens[-1] == self.s_token:
            split_tokens = split_tokens[:-1]
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
            return token_ids_0
        return token_ids_0 + token_ids_1

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
            return offset_mapping_0

        return offset_mapping_0 + offset_mapping_1

    def gen_encode(self, sources, max_src_len=1024, return_tensors=True):
        """
        Main method for encoding the source for generation. It will return a
        dictionary containing the encoded sequence and other relative informations
        which meets the input format requirements of the UNIMO-text model.

        Args:
            source (str): The source text of generation. It should be a string.
            target (str, optional): The target text of generation. It should be
                set when training the model and should be None when running
                inference. Defaults to None.
            title (str, optional): The additional information of some of the
                generation tasks such as summary. Defaults to None.
            max_seq_len (int, optional): The maximum encoded sequence length.
                Defaults to 512.
            max_target_len (int, optional): The maximum encoded sequence
                length of the input `target`. Defaults to 128.
            max_title_len (int, optional): The maximum encoded sequence
                length of the input `title`. Defaults to 128.
            return_position_ids (bool, optional): Whether to return the
                position_ids. Defaults to True.
            return_token_type_ids (bool, optional): Whether to return the
                token_type_ids. Defaults to True.
            return_attention_mask (bool, optional): Whether to return the
                attention_mask. Defaults to True.
            return_length (bool, optional): Whether to return the length of the
                encoded sequence. Defaults to False.
            add_start_token_for_decoding (bool, optional): Whether to add the
                special token "[CLS]" at the end of sequence as the begining of
                the target when running inference to force the model to start
                generating target sequence. Defaults to False.
            pad_to_max_seq_len (bool, optional): Whether to pad the returned
                sequences to the `max_seq_len`. Note that, in this method,
                returned sequences will be padded on the left. Defaults to False.
            return_tensors (bool, optional): Whether to convert the returned
                sequences to Tensor. Defaults to False.
            is_split_into_words(bool, optinal): Whether or not the input text
                (`source`, `target` and `title`) has been pretokenized.
                Defaults to False.
            continuous_position(bool, optinal): Whether the position ids is
                continuous between source ids and target ids. Defaults to False.

        Returns:
            dict: A dictionary containing the encoded sequence and other
            relative informations.

            With the corresponding fields:

            - input_ids (list[int]|Tensor):
                A list of indices of input tokens to be feed to UNIMO-text
                model. If `return_tensors` is True, it is a Tensor with shape
                [1, sequence_length] and data type 'int64'.
            - token_type_ids (list[int]|Tensor, optional):
                A list of segment token indices to indicate whether the token
                belongs to the dialogue target. If `return_tensors` is True,
                it is a Tensor with shape [1, sequence_length] and data type
                'int64'.
                Being returned when `return_token_type_ids` is set to True.
            - position_ids (list[int]|Tensor, optional):
                A list of The position indices. If `return_tensors` is True,
                it is a Tensor with shape [1, sequence_length] and data type
                'int64'.
                Being returned when `return_position_ids` is set to True.
            - attention_mask (numpy.ndarray|Tensor, optional):
                A numpy.ndarray to prevents attention to some unwanted positions,
                with shape [sequence_length, sequence_length] and data type
                'float32'. If `return_tensors` is True, it is a Tensor with shape
                [1, 1, sequence_length, sequence_length] and data type 'float32'.
                Being returned when `return_attention_mask` is set to True.
            - seq_len (int, optional):
                The actual length of the `input_ids`, excluding the pad token.
                Being returned when `return_length` is set to True.

        Example:
            .. code-block::

                from paddlenlp.transformers import UNIMOTokenizer
                tokenizer = UNIMOTokenizer.from_pretrained('unimo-text-1.0')
                inputs = tokenizer.gen_encode('He was a puppeteer')
                #{'input_ids': [1, 4444, 4385, 1545, 6712, 10062, 9568, 9756, 9500, 2],
                #'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                #'position_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                #'attention_mask': array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)}
        """

        def truncate_seq_pair(tokens_a, tokens_b, max_length):
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        # Input type checking for clearer error
        assert isinstance(sources, list), (
            "The input `source` must be with type `list`. "
            " But received: {}".format(sources))
        assert isinstance(sources[0], str), (
            "The element in  `source` must be with type `str`. "
            " But received: {}".format(sources[0]))
        all_token_ids = []
        all_position_ids = []
        all_pos_ids_extra = []
        all_attention_mask = []

        max_len = 0
        for source in sources:
            if self.mask_token not in source:
                source += self.mask_token
            text_l, text_r = source.split(self.mask_token)
            tokens_l = self._tokenize(text_l)
            tokens_r = self._tokenize(text_r)
            truncate_seq_pair(tokens_l, tokens_r, max_src_len - 1)
            tokens_src = tokens_l + [self.gmask_token] + tokens_r
            tokens_tgt = [self.start_token]
            tokens = tokens_src + tokens_tgt
            max_len = max(len(tokens), max_len)
            token_ids = self.convert_tokens_to_ids(tokens)
            all_token_ids.append(token_ids)
            pos_ids = list(range(0, len(tokens_src))) + [
                tokens_src.index(self.gmask_token)
            ] * len(tokens_tgt)
            all_position_ids.append(pos_ids)
            pos_ids_extra = [0] * len(tokens_src) + list(
                range(1, len(tokens_tgt) + 1))
            all_pos_ids_extra.append(pos_ids_extra)
            sequence_length = len(tokens)
            attention_mask = np.ones(
                (sequence_length, sequence_length), dtype='float32') * -1e4
            start = len(tokens_src)
            end = sequence_length
            attention_mask[:end, :start] = 0.0
            tmp = np.triu(
                np.ones(
                    [end - start, end - start], dtype='float32') * -1e4, 1)
            attention_mask[start:end, start:end] = tmp
            all_attention_mask.append(attention_mask)

        input_token_ids = []
        input_pos_ids = []
        input_pos_ids_extra = []
        input_attention_mask = []
        # Considering that the logits at the last time step in the API of 
        # generative task are taken to generate the next token. In order to 
        # avoid the last time step being a pad, so take padding on the left.
        for token_ids, pos_ids, pos_ids_extra, attention_mask in zip(
                all_token_ids, all_position_ids, all_pos_ids_extra,
                all_attention_mask):
            sequence_length = len(token_ids)
            pad_len = max_len - sequence_length

            token_ids = [self.pad_token_id] * pad_len + token_ids
            input_token_ids.append(token_ids)

            pos_ids = [self.pad_token_id] * pad_len + pos_ids
            input_pos_ids.append(pos_ids)

            pos_ids_extra = [self.pad_token_id] * pad_len + pos_ids_extra
            input_pos_ids_extra.append(pos_ids_extra)

            new_mask = np.ones((max_len, max_len), dtype='float32') * -1e4
            new_mask[-sequence_length:, -sequence_length:] = attention_mask
            input_attention_mask.append(new_mask)

        encoded_inputs = {}
        encoded_inputs["input_ids"] = np.asarray(input_token_ids)
        encoded_inputs["position_ids"] = np.asarray(input_pos_ids)
        encoded_inputs["pos_ids_extra"] = np.asarray(input_pos_ids_extra)
        encoded_inputs["attention_mask"] = np.asarray(input_attention_mask)
        if return_tensors:
            # Add dimentions for batch_size and num_heads
            for k, v in encoded_inputs.items():
                if k == "attention_mask":
                    encoded_inputs[k] = paddle.to_tensor(v).unsqueeze(1)
                else:
                    encoded_inputs[k] = paddle.to_tensor(v)
        return encoded_inputs
