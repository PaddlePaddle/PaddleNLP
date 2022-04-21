#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
from typing import Optional, Union, List

from paddle import Tensor
from paddlenlp.transformers import BertTokenizer

logger = logging.getLogger(__name__)


class T5BertTokenizer(BertTokenizer):
    r"""
    Construct a BERT tokenizer for T5. Based on WordPiece.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self,
                 vocab_file,
                 do_lower_case=False,
                 do_basic_tokenize=True,
                 never_split=None,
                 unk_token="<unk>",
                 sep_token=None,
                 pad_token="<pad>",
                 cls_token=None,
                 mask_token=None,
                 space_token="<space>",
                 tokenize_chinese_chars=True,
                 strip_accents=None,
                 **kwargs):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs, )

        self._space_token = space_token

    def get_vocab(self):
        vocab = {
            self.convert_ids_to_tokens(i): i
            for i in range(self.vocab_size)
        }
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            logging.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int],
            token_ids_1: Optional[List[int]]=None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: ``X </s>``
        - pair of sequences: ``A </s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def tokenize(self, text):
        import re
        # Remove space between <extra_id_*> <spot> <asoc>
        split_bracket = re.compile(
            r"\s*<extra_id_\d>\s*|\s*<spot>\s*|\s*<asoc>\s*")

        if len(split_bracket.split(text)) > 1:
            new_text_list = [split_bracket.split(text)[0]]
            for item in zip(
                    split_bracket.findall(text), split_bracket.split(text)[1:]):
                new_text_list += [item[0].strip(), item[1]]
            text = "".join(new_text_list)
        text = text.replace(' ', self._space_token)
        return super().tokenize(text)

    def _decode(self,
                token_ids: Union[List[int], Tensor],
                skip_special_tokens: bool=False,
                **kwargs) -> str:
        if isinstance(token_ids, Tensor):
            tokens = self.convert_ids_to_tokens(
                token_ids.tolist(), skip_special_tokens=skip_special_tokens)
        else:
            tokens = self.convert_ids_to_tokens(
                token_ids, skip_special_tokens=skip_special_tokens)

        # Fix '##' subtoken
        tokens = [x.lstrip('#') if x.startswith("##") else x for x in tokens]

        x_str = "".join(tokens)
        x_str = x_str.replace(' ', '')
        x_str = x_str.replace(self._space_token, ' ')
        return x_str

    def decode(self,
               token_ids: Union[List[int], Tensor],
               skip_special_tokens: bool=False,
               **kwargs) -> str:
        return self._decode(token_ids, skip_special_tokens)

    def batch_decode(self,
                     sequences,
                     skip_special_tokens=False,
                     clean_up_tokenization_spaces=True):
        """
        Convert a list of lists of token ids into a list of strings by calling decode.
        Args:
            sequences (Union[List[int], List[List[int]], Tensor]):
                List of tokenized input ids.
            skip_special_tokens (bool, optional):
                Whether or not to remove special tokens in the decoding. Defaults to `False`.
            clean_up_tokenization_spaces (bool, optional):
                Whether or not to clean up the tokenization spaces. Defaults to `True`.
        Returns:
            List[str]: The list of decoded sentences.
        """
        return [
            self._decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces)
            for seq in sequences
        ]
