# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

__all__ = [
    "ArtistTokenizer",
]


class ArtistTokenizer(BertTokenizer):
    """
    Constructs an Artist tokenizer. `ArtistTokenizer` is almost identical to `BertTokenizer`.

    Args:
        vocab_file (str):
            The vocabulary file path (ends with '.txt') required to instantiate
            a `WordpieceTokenizer`.
        do_lower_case (bool, optional):
            Whether to lowercase the input when tokenizing.
            Defaults to `True`.
        image_vocab_size (int, optional):
            The vocabulary size of image.
            Defaults to `16384`.
        do_basic_tokenize (bool, optional):
            Whether to use a basic tokenizer before a WordPiece tokenizer.
            Defaults to `True`.
        never_split (Iterable, optional):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`. Defaults to `None`.
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
        tokenize_chinese_chars (bool, optional):
            Whether to tokenize Chinese characters.
            Defaults to `True`.
        strip_accents: (bool, optional):
            Whether to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
            Defaults to `None`.

    Examples:
        .. code-block::

            from paddlenlp.transformers import ArtistTokenizer
            tokenizer = ArtistTokenizer.from_pretrained('pai-painter-painting-base-zh')

            inputs = tokenizer('风阁水帘今在眼，且来先看早梅红', return_token_type_ids=False)
            print(inputs)

            '''
            {'input_ids': [23983, 23707, 20101, 18750, 17175, 18146, 21090, 24408, 17068,
                           19725, 17428, 21076, 19577, 19833, 21657]}
            '''

    """

    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "pai-painter-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-base-zh/vocab.txt",
            "pai-painter-painting-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-painting-base-zh/vocab.txt",
            "pai-painter-scenery-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-scenery-base-zh/vocab.txt",
            "pai-painter-commercial-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-commercial-base-zh/vocab.txt",
            "pai-painter-large-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-large-zh/vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "pai-painter-base-zh": {
            "do_lower_case": True,
            "image_vocab_size": 16384,
        },
        "pai-painter-painting-base-zh": {
            "do_lower_case": True,
            "image_vocab_size": 16384,
        },
        "pai-painter-scenery-base-zh": {
            "do_lower_case": True,
            "image_vocab_size": 16384,
        },
        "pai-painter-commercial-base-zh": {
            "do_lower_case": True,
            "image_vocab_size": 16384,
        },
        "pai-painter-large-zh": {
            "do_lower_case": True,
            "image_vocab_size": 16384,
        },
    }
    max_model_input_sizes = {
        "pai-painter-base-zh": 32,
        "pai-painter-painting-base-zh": 32,
        "pai-painter-scenery-base-zh": 32,
        "pai-painter-commercial-base-zh": 32,
        "pai-painter-large-zh": 32,
    }

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        image_vocab_size=16384,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            do_lower_case,
            do_basic_tokenize,
            never_split,
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token,
            tokenize_chinese_chars,
            strip_accents,
            **kwargs,
        )
        # we need add image_vocab_size offset
        # for example [523, 102, 0, 0]
        # => [523 + image_vocab_size, 102 + image_vocab_size, 0 + image_vocab_size, 0 + image_vocab_size]
        self.image_vocab_size = image_vocab_size

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            # note: process image_vocab_size offset
            return self.added_tokens_encoder[token] + self.image_vocab_size
        # note: process image_vocab_size offset
        return self._convert_token_to_id(token) + self.image_vocab_size

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                # note: process image_vocab_size offset
                return self._convert_id_to_token(ids - self.image_vocab_size)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                # note: process image_vocab_size offset
                tokens.append(self._convert_id_to_token(index - self.image_vocab_size))
        return tokens

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence (we don't add special tokens).

        An Artist sequence has the following format:

        - single sequence:      ``X``

        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.
                We do'nt use sequence pairs.
                Defaults to None.

        Returns:
            List[int]: List of input_id.
        """
        return token_ids_0

    def __call__(
        self,
        text,
        text_pair=None,
        max_length=32,  # default
        stride=0,
        is_split_into_words=False,
        padding="max_length",  # default
        truncation=True,  # default
        return_position_ids=False,
        return_token_type_ids=False,  # don't return token_type_ids
        return_attention_mask=False,
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
