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
import itertools
from paddle.utils import try_import
from .. import GPTTokenizer

__all__ = ['BartTokenizer']


class BartTokenizer(GPTTokenizer):
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
        special_tokens (list, optional):
            A list of special tokens not in the vocabulary.
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
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt"
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "bart-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bart/bart-base-vocab.json",
            "bart-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bart/bart-large-vocab.json",
        },
        "merges_file": {
            "bart-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bart/bart-base-merges.txt",
            "bart-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bart/bart-large-merges.txt",
        }
    }
    pretrained_init_configuration = {"bart-base": {}, "bart-large": {}}

    def __init__(
            self,
            vocab_file,
            merges_file,
            errors='replace',
            max_len=None,
            special_tokens=None,
            bos_token="<s>",
            eos_token="</s>",
            cls_token="<s>",
            sep_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>", ):
        super(BartTokenizer, self).__init__(vocab_file, merges_file, errors,
                                            max_len, special_tokens, pad_token,
                                            eos_token)

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
                 return_attention_mask=True,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        return super(BartTokenizer, self).__call__(
            text, text_pair, max_seq_len, stride, is_split_into_words,
            pad_to_max_seq_len, truncation_strategy, return_position_ids,
            return_token_type_ids, return_attention_mask, return_length,
            return_overflowing_tokens, return_special_tokens_mask)

    def _bpe_encode(self, text):
        bpe_tokens = []
        re = try_import("regex")
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(
                bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def _tokenize(self, text):
        """ Tokenize a string. """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if tok == self.mask_token and i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            """
            Process special tokens.
            Don't split special tokens when in Mask Filling task.
            """
            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.all_special_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable((self._bpe_encode(
                    token) if token not in self.all_special_tokens else [
                        token
                    ] for token in tokenized_text)))

        tokenized_text = split_on_tokens(self.all_special_tokens, text)
        return tokenized_text

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

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is
        called when adding special tokens using the tokenizer ``encode`` methods.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True)
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)
                                                          ) + [1]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
