# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Dict, List, Tuple

from fast_tokenizer import Tokenizer, normalizers, pretokenizers, postprocessors, decoders
from fast_tokenizer.models import WordPiece, FastWordPiece, BPE, Unigram


# Extract the vocab and merge file from sentencepiece file
class SentencePieceExtractor:
    def __init__(self, model: str):
        from sentencepiece import SentencePieceProcessor

        self.sp = SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self):
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}

        # Merges
        merges = []
        for piece_l in vocab.keys():
            for piece_r in vocab.keys():
                merge = f"{piece_l}{piece_r}"
                piece_id = vocab.get(merge, None)
                if piece_id:
                    merges += [(piece_l, piece_r, piece_id)]
        merges = sorted(merges, key=lambda val: val[2])
        merges = [(val[0], val[1]) for val in merges]

        return vocab, merges


def check_number_comma(piece: str) -> bool:
    return len(piece) < 2 or piece[-1] != "," or not piece[-2].isdigit()


class Converter:
    def __init__(self, original_tokenizer):
        self.original_tokenizer = original_tokenizer

    def converted(self) -> Tokenizer:
        raise NotImplementedError()


class BertConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(
            FastWordPiece(
                vocab._token_to_idx, unk_token=str(self.original_tokenizer.unk_token), with_pretokenization=True
            )
        )

        tokenize_chinese_chars = True
        strip_accents = True
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        # No need to init pretokenizer because FastWordPiece can pretokenize
        # tokenizer.pretokenizer = pretokenizers.BertPreTokenizer()

        cls_token = str(self.original_tokenizer.cls_token)
        sep_token = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.postprocessor = postprocessors.BertPostProcessor(
            (str(sep_token), sep_token_id), (str(cls_token), cls_token_id)
        )

        tokenizer.decoder = decoders.WordPiece(prefix="##")
        return tokenizer


class ErnieConverter(BertConverter):
    pass


class TinyBertConverter(BertConverter):
    pass


# For sentencepiece tokenzier
class SpmConverter(Converter):
    def __init__(self, *args):
        super().__init__(*args)
        from . import sentencepiece_model_pb2 as model_pb2

        m = model_pb2.ModelProto()
        # For ernie_m sentencepiece tokenizer
        if hasattr(self.original_tokenizer, "sentencepiece_model_file"):
            spm_vocab_file = self.original_tokenizer.sentencepiece_model_file
        else:
            spm_vocab_file = self.original_tokenizer.vocab_file
        with open(spm_vocab_file, "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m

    def vocab(self, proto):
        return [(piece.piece, piece.score) for piece in proto.pieces]

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab = self.vocab(proto)
        unk_id = self.unk_id(proto)
        if model_type == 1:
            tokenizer = Tokenizer(Unigram(vocab, unk_id))
        elif model_type == 2:
            # Special case for ernie-m
            if hasattr(self.original_tokenizer, "sentencepiece_model_file"):
                orginal_vocab_file = self.original_tokenizer.sentencepiece_model_file
            else:
                orginal_vocab_file = self.original_tokenizer.vocab_file
            _, merges = SentencePieceExtractor(orginal_vocab_file).extract()
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab)}
            tokenizer = Tokenizer(
                BPE(
                    bpe_vocab,
                    merges,
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                )
            )
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        if not precompiled_charsmap:
            return normalizers.SequenceNormalizer([normalizers.ReplaceNormalizer(" {2,}", " ")])
        else:
            return normalizers.SequenceNormalizer(
                [normalizers.PrecompiledNormalizer(precompiled_charsmap), normalizers.ReplaceNormalizer(" {2,}", " ")]
            )

    def pretokenizer(self, replacement, add_prefix_space):
        return pretokenizers.MetaSpacePreTokenizer(replacement=replacement, add_prefix_space=add_prefix_space)

    def postprocessor(self):
        return None

    def replacement(self):
        return "▁"

    def add_prefix_space(self):
        return True

    def set_model(self, tokenizer):
        pass

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        self.set_model(tokenizer)
        # Tokenizer assemble
        tokenizer.normalizer = self.normalizer(self.proto)

        replacement = self.replacement()
        add_prefix_space = self.add_prefix_space()
        tokenizer.pretokenizer = self.pretokenizer(replacement, add_prefix_space)
        # tokenizer.decoder = decoders.MetaSpace(replacement=replacement, add_prefix_space=add_prefix_space)
        postprocessor = self.postprocessor()
        if postprocessor:
            tokenizer.postprocessor = postprocessor

        return tokenizer


class ErnieMConverter(SpmConverter):
    def set_model(self, tokenizer):
        SPLICE_UNDERLINE = self.replacement()
        tokenizer.model.set_filter_token(SPLICE_UNDERLINE)
        chinese_chars = r"\x{4e00}-\x{9fff}"
        punc_chars = r",;:.?!~，；：。？！《》【】"
        digits = r"0-9"
        tokenizer.model.set_split_rule(
            rf"[{chinese_chars}]|[{punc_chars}]|[{digits}]+|[^{chinese_chars}{punc_chars}{digits}]+"
        )

    def normalizer(self, proto):
        list_normalizers = []
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        list_normalizers.append(normalizers.PrecompiledNormalizer(precompiled_charsmap))
        return normalizers.SequenceNormalizer(list_normalizers)

    def vocab(self, proto):
        # construct a dict that map word and score
        word_score_dict = {}
        for piece in proto.pieces:
            word_score_dict[piece.piece] = piece.score
        vocab_list = [None] * len(self.original_tokenizer.vocab)
        original_vocab = self.original_tokenizer.vocab.token_to_idx
        for _token, _id in original_vocab.items():
            if _token in word_score_dict:
                vocab_list[_id] = (_token, word_score_dict[_token])
            else:
                vocab_list[_id] = (_token, 0.0)
        return vocab_list

    def unk_id(self, proto):
        return self.original_tokenizer.convert_tokens_to_ids(str(self.original_tokenizer.unk_token))

    def pretokenizer(self, replacement, add_prefix_space):
        return pretokenizers.SequencePreTokenizer(
            [
                pretokenizers.WhitespacePreTokenizer(),
                pretokenizers.MetaSpacePreTokenizer(replacement=replacement, add_prefix_space=add_prefix_space),
            ]
        )

    def postprocessor(self):
        """
         An ERNIE-M sequence has the following format:
        - single sequence:       ``[CLS] X [SEP]``
        - pair of sequences:        ``[CLS] A [SEP] [SEP] B [SEP]``
        """
        return postprocessors.TemplatePostProcessor(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 [SEP]:1 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


SLOW_TO_FAST_CONVERTERS = {
    "BertTokenizer": BertConverter,
    "ErnieTokenizer": ErnieConverter,
    "TinyBertTokenizer": TinyBertConverter,
    "ErnieMTokenizer": ErnieMConverter
    # TODO(zhoushunjie): Need to implement more TokenizerConverter
}


def convert_slow_tokenizer(transformer_tokenizer) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer ([`~tokenizer_utils_base.PretrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenizer_utils_base.PretrainedFastTokenizer`].

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenizer_utils_base.PretrainedFastTokenizer`]
    """

    tokenizer_class_name = transformer_tokenizer.__class__.__name__

    if tokenizer_class_name not in SLOW_TO_FAST_CONVERTERS:
        raise ValueError(
            f"An instance of tokenizer class {tokenizer_class_name} cannot be converted in a Fast tokenizer instance. "
            f"No converter was found. Currently available slow->fast convertors: {list(SLOW_TO_FAST_CONVERTERS.keys())}"
        )

    converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]

    return converter_class(transformer_tokenizer).converted()
