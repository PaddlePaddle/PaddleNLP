# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Any, Dict

import icetk.sentencepiece_model_pb2 as icetk_model
import paddle
import paddle.nn.functional as F
from icetk.text_tokenizer import TextTokenizer
from paddle import Tensor

from paddlenlp.transformers import PretrainedTokenizer
from paddlenlp.transformers.tokenizer_utils_base import BatchEncoding

__all__ = ["GLM130BTokenizer"]


class GLM130BTokenizer(PretrainedTokenizer):
    resource_files_names = {"model_file": "ice_text.model"}
    pretrained_resource_files_map = {
        "model_file": {
            "THUDM/glm-130b": "https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-130b/ice_text.model"
        },
        # "added_tokens_file": {
        #    "THUDM/glm-130b": "https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-130b/added_tokens.json"
        # },
    }
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {"glm-130b": {}}
    max_model_input_sizes = {"THUDM/glm-130b": 2048}

    def __init__(
        self,
        model_file,
        max_blank_length=80,
        byte_fallback=True,
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="[ENC]",
        mask_token="[MASK]",
        **kwargs
    ):
        # additional_special_tokens = [
        #    "[MASK]", "[gMASK]", "[sMASK]", "eod", "sop", "eop", "<ENC>", "<dBLOCK>",
        #    "<t>", *[f"<|blank_{i}|>" for i in range(2, max_blank_length + 1)]
        # ]
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            # additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        if not os.path.isfile(model_file):
            raise ValueError(
                f"Can't find a vocabulary model file at path {model_file}. "
                "To load the vocabulary from pretrained model please use "
                "`tokenizer = GLM130BTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.model_file = model_file
        self.max_blank_length = max_blank_length
        self.byte_fallback = byte_fallback
        self.sentence_end_decoder = {
            20007: ".",
            20031: "？",
            20035: "！",
            20027: "；",
            20012: ":",
            83823: "。",
            145670: "…",
        }
        self.added_tokens_encoder.update({"eos": 20002})
        self.added_tokens_decoder.update({20002: "</s>"})

        # TODO The version may be unstable, since special tokens are defined repeatedly in sp_model.
        self.smp_special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "<eod>", "<sop>", "<eop>", "<ENC>", "<dBLOCK>"]

    @property
    def text_tokenizer(self):
        return self._get_text_tokenizer(encode_special_tokens=False)

    @property
    def special_text_tokenizer(self):
        return self._get_text_tokenizer(encode_special_tokens=True)

    def _get_text_tokenizer(self, encode_special_tokens=False):
        name = "_special_text_tokenizer" if encode_special_tokens else "_text_tokenizer"
        if not hasattr(self, name):
            tokenizer = TextTokenizer(self.model_file)
            self._configure_tokenizer(tokenizer, encode_special_tokens)
            setattr(self, name, tokenizer)
        return getattr(self, name)

    def _configure_tokenizer(self, text_tokenizer, encode_special_tokens=False):
        # special token
        special_token_type = 4 if encode_special_tokens else 3  # 3 - CONTROL, 4 - USER_DEFINE
        for token in self.smp_special_tokens:
            text_tokenizer.proto.pieces.append(
                icetk_model.ModelProto.SentencePiece(piece=token, score=0.0, type=special_token_type)
            )
        # whitespaces
        for token in [self.tab_token] + [self.get_blank_token(i) for i in range(2, self.max_blank_length + 1)]:
            text_tokenizer.proto.pieces.append(icetk_model.ModelProto.SentencePiece(piece=token, score=0.0, type=4))
        # byte fallback
        if self.byte_fallback:
            text_tokenizer.proto.trainer_spec.byte_fallback = True
            for i in range(256):
                text_tokenizer.proto.pieces.append(
                    icetk_model.ModelProto.SentencePiece(piece="<0x{:02X}>".format(i), score=0.0, type=6)
                )
        text_tokenizer.refresh()

    def get_vocab(self):
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    @property
    def sop_token(self):
        return "<sop>"

    @property
    def sop_token_id(self):
        return self.convert_tokens_to_ids(self.sop_token)

    @property
    def eop_token(self):
        return "<eop>"

    @property
    def eop_token_id(self):
        return self.convert_tokens_to_ids(self.eop_token)

    @property
    def eod_token(self):
        return "<eod>"

    @property
    def eod_token_id(self):
        return self.convert_tokens_to_ids(self.eod_token)

    @property
    def gmask_token(self):
        return "[gMASK]"

    @property
    def gmask_token_id(self):
        return self.convert_tokens_to_ids(self.gmask_token)

    @property
    def smask_token(self):
        return "[sMASK]"

    @property
    def smask_token_id(self):
        return self.convert_tokens_to_ids(self.smask_token)

    @property
    def tab_token(self):
        return "<|tab|>"

    @property
    def num_image_tokens(self):
        return 20000

    @property
    def num_text_tokens(self):
        return (
            self.text_tokenizer.num_tokens
            + len(self.smp_special_tokens)
            + (self.max_blank_length - 2)
            + (256 if self.byte_fallback else 0)
        )

    @property
    def vocab_size(self):
        return self.num_text_tokens + self.num_image_tokens + len(self.added_tokens_encoder)

    def get_blank_token(self, length: int):
        assert length >= 2 and length <= self.max_blank_length
        return f"<|blank_{length}|>"

    def contains_sentence_end(self, index):
        return index in self.sentence_end_decoder

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # TODO Missing branch to deal with inputs with is_split_into_words=True.
        linebreak = kwargs.get("linebreak", True)
        if linebreak:
            text = text.replace("\n", "<n>")

        whitespace = kwargs.get("whitespace", True)
        if whitespace:
            text = text.replace("\t", self.tab_token)
            for i in range(self.max_blank_length, 1, -1):
                text = text.replace(" " * i, self.get_blank_token(i))

        add_dummy_prefix = kwargs.get("add_dummy_prefix", True)
        if not add_dummy_prefix:
            text = "<n>" + text

        return text, kwargs

    def _convert_token_to_id(self, token):
        if token.startswith("<image_") and token.endswith(">") and token[7:-1].isdigit():
            return int(token[7:-1])
        else:
            return self.text_tokenizer.convert_token_to_id(token) + self.num_image_tokens

    def _convert_id_to_token(self, index):
        if index < self.num_image_tokens:
            return f"<image_{index}>"
        else:
            return self.text_tokenizer.convert_id_to_token(index - self.num_image_tokens)

    def convert_tokens_to_strings(self, tokens):
        return "".join(tokens)

    def _tokenize(self, text):
        return self.text_tokenizer.tokenize(text)

    def tokenize(self, text, **kwargs):
        tokenized_text = super().tokenize(text, **kwargs)
        add_dummy_prefix = kwargs.get("add_dummy_prefix", True)
        return tokenized_text if add_dummy_prefix else tokenized_text[2:]

    def decode(
        self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True, **kwargs
    ) -> str:
        text = super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        text = text.replace("<n>", "\n")
        text = text.replace(self.tab_token, "\t")
        for i in range(2, self.max_blank_length + 1):
            text = text.replace(self.get_blank_token(i), " " * i)
        return text

    def _build_inputs_for_generation(self, model_input: Dict, max_length: int = 256, padding: bool = False):
        input_ids = model_input["input_ids"]
        input_ids = input_ids[input_ids > 0]
        if isinstance(input_ids, Tensor):
            input_ids = input_ids.tolist()
        generation_mask = self.gmask_token_id
        if self.mask_token_id in input_ids:
            generation_mask = self.mask_token_id
        elif self.smask_token_id in input_ids:
            generation_mask = self.smask_token_id
        use_gmask = generation_mask == self.gmask_token_id

        if (self.mask_token_id in input_ids) + (self.smask_token_id in input_ids) + (
            self.gmask_token_id in input_ids
        ) == 0:
            input_ids += [generation_mask]
        if (input_ids[-1] == self.mask_token_id) + (input_ids[-1] == self.smask_token_id) + (
            input_ids[-1] == self.gmask_token_id
        ) == 0:
            input_ids += [self.eos_token_id]
        input_ids += [self.sop_token_id]
        context_length = len(input_ids)

        if len(input_ids) > max_length:
            raise ValueError(f"Input sequence length {len(input_ids)} exceeds maximum length {max_length}.")

        input_ids = paddle.to_tensor(input_ids, dtype="int64")
        if padding:
            input_ids = F.pad(input_ids, [0, max_length - len(input_ids)], mode="constant", value=-1)

        mask_position = paddle.where(input_ids == generation_mask)[0].tolist()[0][0]

        attention_mask = paddle.ones([1, input_ids.shape[-1], input_ids.shape[-1]], dtype="float32")
        attention_mask = paddle.tril(attention_mask)
        attention_mask[..., : context_length - 1] = 1
        attention_mask = (attention_mask < 0.5).astype("int64")

        position_ids = paddle.arange(input_ids.shape[-1], dtype="int64")
        if not use_gmask:
            position_ids[context_length - 1 :] = mask_position

        return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

    def build_inputs_for_generation(
        self,
        model_input: BatchEncoding,
        max_length: int = 256,
        padding=False,
    ):
        samples = [{key: value[i] for key, value in model_input.items()} for i in range(len(model_input["input_ids"]))]
        samples = [self._build_inputs_for_generation(sample, max_length, padding) for sample in samples]
        inputs = self._collate(samples)
        return BatchEncoding(inputs)

    def _collate(self, samples):
        length_to_pad = max([len(sample["input_ids"]) for sample in samples])

        input_ids_batch = []
        position_ids_batch = []
        attention_mask_batch = []
        for sample in samples:
            pad_length = length_to_pad - len(sample["input_ids"])
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]
            position_ids = sample["position_ids"]
            if pad_length > 0:
                input_ids = F.pad(input_ids, [0, pad_length], mode="constant", value=-1)
                attention_mask = F.pad(attention_mask, [0, 0, 0, pad_length, 0, pad_length], mode="constant", value=0)
                position_ids = paddle.concat([position_ids, position_ids[-1:].expand([pad_length])], axis=-1)
            input_ids_batch.append(input_ids)
            position_ids_batch.append(position_ids)
            attention_mask_batch.append(attention_mask)
        return BatchEncoding(
            {
                "input_ids": paddle.stack(input_ids_batch, axis=0),
                "position_ids": paddle.stack(position_ids_batch, axis=0),
                "attention_mask": paddle.stack(attention_mask_batch, axis=0),
            }
        )

    def is_english(self, text):
        try:
            text.encode(encoding="utf-8").decode("ascii")
        except UnicodeDecodeError:
            return False
        else:
            return True
