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

from typing import Dict, List, Optional, Tuple

from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    normalizers,
    pre_tokenizers,
    processors,
)
from tokenizers.models import BPE, Unigram, WordPiece


# Extract the vocab and merge file from sentencepiece file
class SentencePieceExtractor:
    def __init__(self, model: str):
        from sentencepiece import SentencePieceProcessor

        self.sp = SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self, vocab_scores: Optional[Tuple[str, float]] = None) -> Tuple[Dict[str, int], List[Tuple]]:
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}
        if vocab_scores is not None:
            vocab_scores, reverse = dict(vocab_scores), True
        else:
            vocab_scores, reverse = vocab, False

        # Merges
        merges = []
        for merge, piece_score in vocab_scores.items():
            local = []
            for index in range(1, len(merge)):
                piece_l, piece_r = merge[:index], merge[index:]
                if piece_l in vocab and piece_r in vocab:
                    local.append((piece_l, piece_r, piece_score))
            local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
            merges.extend(local)

        merges = sorted(merges, key=lambda val: val[2], reverse=reverse)
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
        tokenizer = Tokenizer(WordPiece(vocab._token_to_idx, unk_token=str(self.original_tokenizer.unk_token)))

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
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls_token = str(self.original_tokenizer.cls_token)
        sep_token = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls_token}:0 $A:0 {sep_token}:0",
            pair=f"{cls_token}:0 $A:0 {sep_token}:0 $B:1 {sep_token}:1",
            special_tokens=[
                (cls_token, cls_token_id),
                (sep_token, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class ErnieConverter(BertConverter):
    pass


class RobertaConverter(Converter):
    def converted(self) -> Tokenizer:
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.RobertaProcessing(
            sep=(ot.sep_token, ot.sep_token_id),
            cls=(ot.cls_token, ot.cls_token_id),
            add_prefix_space=ot.add_prefix_space,
            trim_offsets=True,  # True by default on Roberta (historical)
        )

        return tokenizer


class TinyBertConverter(BertConverter):
    pass


class TikTokenConverter(Converter):
    def extract(self, tiktoken_file: str):
        from .tiktoken_model_utils import bpe, bytes_to_unicode, load_tiktoken_bpe

        bpe_ranks = (
            self.original_tokenizer.mergeable_ranks
            if hasattr(self.original_tokenizer, "mergeable_ranks") and self.original_tokenizer.mergeable_ranks
            else load_tiktoken_bpe(tiktoken_file)
        )
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(b):
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        merges = []
        vocab = {}
        for token, rank in bpe_ranks.items():
            vocab[token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = tuple(bpe(bpe_ranks, token, max_rank=rank))
            if len(merged) == 2:
                merges.append(tuple(map(token_bytes_to_string, merged)))

        return vocab, merges


class NystromformerConverter(BertConverter):
    pass


class QWenConverter(TikTokenConverter):
    def converted(self) -> Tokenizer:
        from .qwen.tokenizer import PAT_STR

        ot = self.original_tokenizer
        bpe_vocab, merges = self.extract(ot.tiktoken_file)
        tokenizer = Tokenizer(
            BPE(
                bpe_vocab,  # ot.mergeable_ranks
                merges,
                dropout=None,
                unk_token=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                byte_fallback=False,
            )
        )

        tokenizer.normalizer = normalizers.NFC()

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(PAT_STR),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=getattr(self.original_tokenizer, "add_prefix_space", False),
                    use_regex=False,
                ),
            ]
        )

        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        return tokenizer


class SpmConverter(Converter):
    def __init__(self, *args):

        super().__init__(*args)

        from . import sentencepiece_model_pb2 as model_pb2

        m = model_pb2.ModelProto()
        if hasattr(self.original_tokenizer, "sentencepiece_model_file"):
            spm_vocab_file = self.original_tokenizer.sentencepiece_model_file
        else:
            spm_vocab_file = self.original_tokenizer.vocab_file
        with open(spm_vocab_file, "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m

        if self.proto.trainer_spec.byte_fallback:
            if not getattr(self, "handle_byte_fallback", None):
                import warnings

                warnings.warn(
                    "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
                    " which is not implemented in the fast tokenizers. In practice this means that the fast version of the"
                    " tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these "
                    "unknown tokens into a sequence of byte tokens matching the original piece of text."
                )

    def vocab(self, proto):
        return [(piece.piece, piece.score) for piece in proto.pieces]

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        unk_id = self.unk_id(proto)

        if model_type == 1:
            tokenizer = Tokenizer(Unigram(vocab_scores, unk_id))
        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract()
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
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
        _normalizers = [
            normalizers.Strip(left=False, right=True),  # stripping is important
            normalizers.Replace(Regex(" {2,}"), "▁"),
        ]
        if not precompiled_charsmap:
            return normalizers.Sequence(_normalizers)
        else:
            return normalizers.Sequence([normalizers.Precompiled(precompiled_charsmap)] + _normalizers)

    def pre_tokenizer(self, replacement, add_prefix_space):
        prepend_scheme = "always"
        if hasattr(self.original_tokenizer, "legacy") and not self.original_tokenizer.legacy:
            prepend_scheme = "first"
        return pre_tokenizers.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space, prepend_scheme=prepend_scheme
        )

    def post_processor(self):
        return None

    def decoder(self, replacement, add_prefix_space):
        return decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        return tokenizer


class AlbertConverter(SpmConverter):
    def vocab(self, proto):
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    def normalizer(self, proto):
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class ErnieMConverter(SpmConverter):
    def normalizer(self, proto):
        list_normalizers = []
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        return normalizers.Sequence(list_normalizers)

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
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.Whitespace(),
                pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
            ]
        )

    def post_processor(self):
        """
         An ERNIE-M sequence has the following format:
        - single sequence:       ``[CLS] X [SEP]``
        - pair of sequences:        ``[CLS] A [SEP] [SEP] B [SEP]``
        """
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 [SEP]:1 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class LlamaConverter(SpmConverter):
    handle_byte_fallback = True

    def vocab(self, proto):
        vocab = [
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        unk_id = 0
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        return decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(content=" ", left=1),
            ]
        )

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        if model_type == 1:
            import tokenizers
            from packaging import version

            if version.parse(tokenizers.__version__) < version.parse("0.14.0"):
                tokenizer = Tokenizer(Unigram(vocab_scores, 0))
            else:
                tokenizer = Tokenizer(Unigram(vocab_scores, 0, byte_fallback=True))

        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(bpe_vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True, byte_fallback=True)
            )
            tokenizer.add_special_tokens(
                [
                    AddedToken("<unk>", normalized=False, special=True),
                    AddedToken("<s>", normalized=False, special=True),
                    AddedToken("</s>", normalized=False, special=True),
                ]
            )
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but your file was trained with a different algorithm"
            )

        return tokenizer

    def normalizer(self, proto):
        return normalizers.Sequence(
            [
                normalizers.Prepend(prepend="▁"),
                normalizers.Replace(pattern=" ", content="▁"),
            ]
        )

    def pre_tokenizer(self, replacement, add_prefix_space):
        return None

    def post_processor(self):
        # the post_processor is defined in LlamaTokenizerFast class.
        return None


class ErnieTinyConverter(ErnieMConverter):
    pass


SLOW_TO_FAST_CONVERTERS = {
    "BertTokenizer": BertConverter,
    "ErnieTokenizer": ErnieConverter,
    "TinyBertTokenizer": TinyBertConverter,
    "ErnieMTokenizer": ErnieMConverter,
    "NystromformerTokenizer": NystromformerConverter,
    "DistilBertTokenizer": BertConverter,
    "AlbertEnglishTokenizer": AlbertConverter,
    # "AlbertEnglishTokenizer": AlbertConverter,
    "RobertaBPETokenizer": RobertaConverter,
    "ErnieTinyTokenizer": ErnieTinyConverter,
    "LlamaTokenizer": LlamaConverter,
    "QWenTokenizer": QWenConverter,
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
