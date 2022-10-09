# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

{%- if cookiecutter.tokenizer_type == "Based on Bert" %}
from ..bert.tokenizer import BertTokenizer


__all__ = [
    '{{cookiecutter.camelcase_modelname}}Tokenizer',
]


class {{cookiecutter.camelcase_modelname}}Tokenizer(PretrainedTokenizer):
    """
    Constructs a {{cookiecutter. lowercase_modelname}} tokenizer Based on BertTokenizer

    Args:
        vocab_file (str):
            The vocabulary file path (ends with '.txt') required to instantiate
            a `WordpieceTokenizer`.
        do_lower_case (bool, optional):
            Whether to lowercase the input when tokenizing.
            Defaults to `True`.
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

            from paddlenlp.transformers import {{cookiecutter.camelcase_modelname}}Tokenizer
            tokenizer = {{cookiecutter.camelcase_modelname}}Tokenizer.from_pretrained('{{cookiecutter.checkpoint_identifier}}')

            inputs = tokenizer('He was a puppeteer')
            print(inputs)
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
        }
    }
    pretrained_init_configuration = {
    }
    max_model_input_sizes = {
    }
    padding_side = 'right'

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 never_split=None,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 tokenize_chinese_chars=True,
                 strip_accents=None,
                 **kwargs):
        super({{cookiecutter.camelcase_modelname}}Tokenizer, self).__init__(
                 vocab_file,
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
                 **kwargs
        )

{%- elif cookiecutter.tokenizer_type == "Based on BPETokenizer" %}

from ..tokenizer_utils import BPETokenizer

class {{cookiecutter.camelcase_modelname}}Tokenizer(BPETokenizer):
    """
    Construct {{cookiecutter.camelcase_modelname}}Tokenizer Based on BPETokenizer.   

    Args:
        encoder_json_path (str, optional):
            file path of the id to vocab.
        vocab_bpe_path (str, optional):
            file path of word merge text.
        unk_token (str, optional): 
            The special token for unknown words. 
            Defaults to "[UNK]".
        sep_token (str, optional): 
            The special token for separator token. 
            Defaults to "[SEP]".
        pad_token (str, optional): 
            The special token for padding. 
            Defaults to "[PAD]".
        cls_token (str, optional): 
            The special token for cls. 
            Defaults to "[CLS]".
        mask_token (str, optional): 
            The special token for mask.
            Defaults to "[MASK]".
    """
    def __init__(self,
                 vocab_file: str,
                 merges_file: str,
                 do_lower_case: bool = False,
                 unk_token: str = "[UNK]",
                 sep_token: str = "[SEP]",
                 pad_token: str = "[PAD]",
                 cls_token: str = "[CLS]",
                 mask_token: str = "[MASK]",
                 **kwargs):
        super({{cookiecutter.camelcase_modelname}}Tokenizer, self).__init__(
            vocab_file=vocab_file ,
            merges_file =merges_file,
            do_lower_case = do_lower_case,
            unk_token = unk_token,
            sep_token = sep_token,
            pad_token = pad_token,
            cls_token = cls_token,
            mask_token = mask_token,
            **kwargs
        )

{%- elif cookiecutter.tokenizer_type == "Based on SentencePiece" %}

class {{cookiecutter.camelcase_modelname}}Tokenizer(PretrainedTokenizer):
    resource_files_names = {
        "sentencepiece_model_file": "spiece.model",
    }

    pretrained_resource_files_map = {
        "sentencepiece_model_file": {
        },
    }

    pretrained_init_configuration = {
    }
    max_model_input_sizes = {
    }

    def __init__(self,
                 sentencepiece_model_file,
                 do_lower_case=True,
                 remove_space=True,
                 keep_accents=False,
                 bos_token="[CLS]",
                 eos_token="[SEP]",
                 unk_token="<unk>",
                 sep_token="[SEP]",
                 pad_token="<pad>",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 sp_model_kwargs=None,
                 **kwargs):

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.sentencepiece_model_file = sentencepiece_model_file

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(sentencepiece_model_file)

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {
            self.convert_ids_to_tokens(i): i
            for i in range(self.vocab_size)
        }
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.sentencepiece_model_file)

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join(
                [c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text):
        """Tokenize a string."""
        text = self.preprocess_text(text)
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(
                    SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][
                        0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def _convert_token_to_id(self, token):
        """Converts a token (str) to an id using the vocab. """
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) to a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(
                token_ids_0, token_ids_1 if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(
                    lambda x: 1
                    if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_resources(self, save_directory):
        for name, file_name in self.resource_files_names.items():
            save_path = os.path.join(save_directory, file_name)
            if os.path.abspath(
                    self.sentencepiece_model_file) != os.path.abspath(
                        save_path) and os.path.isfile(
                            self.sentencepiece_model_file):
                copyfile(self.sentencepiece_model_file, save_path)
            elif not os.path.isfile(self.sentencepiece_model_file):
                with open(save_path, "wb") as fi:
                    content_spiece_model = self.sp_model.serialized_model_proto(
                    )
                    fi.write(content_spiece_model)

{% endif %}
