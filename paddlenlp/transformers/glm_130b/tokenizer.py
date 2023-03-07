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
import shutil
from typing import Any, Dict, List

import paddle
import paddle.nn.functional as F
import sentencepiece as spm
from paddle import Tensor

from .. import PretrainedTokenizer
from ..tokenizer_utils_base import BatchEncoding

__all__ = ["GLM130BTokenizer"]


class GLM130BTokenizer(PretrainedTokenizer):
    resource_files_names = {"model_file": "ice_text.model"}
    pretrained_resource_files_map = {
        "model_file": {"glm-130b": "https://paddlenlp.bj.bcebos.com/models/transformers/glm-130b/ice_text.model"}
    }
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {"glm-130b": {}}

    def __init__(self, model_file, max_len=512, unk_token="<unk>", ignore_linebreak=True, **kwargs):
        self._model_file = model_file
        self.unk_token = unk_token
        self.pad_token = "[pad]"
        self.ignore_linebreak = ignore_linebreak
        if not os.path.isfile(model_file):
            raise ValueError(
                "Can't find a model file at path '{}'. To load the "
                "model from a pretrained model please use "
                "`tokenizer = GLM130BTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(model_file)
            )
        self.max_len = max_len if max_len is not None else int(1e12)
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_file)
        self.num_image_tokens = 20000
        self.num_tokens = 150000
        self.add_tokens(["[MASK]", "[gMASK]", "[sMASK]", "[eod]", "[sop]", "[eop]", "[ENC]", "[dBLOCK]"])
        self.sentence_end_decoder = {
            20007: ".",
            20031: "？",
            20035: "！",
            20027: "；",
            20012: ":",
            83823: "。",
            145670: "…",
        }
        self.added_tokens_encoder["[eos]"] = 20002
        self.added_tokens_decoder[20002] = "</s>"
        self.added_tokens_encoder["[pad]"] = 0
        self.added_tokens_decoder[0] = "[pad]"

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

    @property
    def vocab_size(self) -> int:
        return self.num_tokens

    def _tokenize(self, text: str) -> List[str]:
        if not self.ignore_linebreak:
            text = text.replace("\n", "<n>")
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        if token == "[pad]":
            return 0
        elif token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self.sp_model.PieceToId(token.replace("\n", "<n>")) + self.num_image_tokens

    def _convert_id_to_token(self, index):
        if index == 0:
            return "[pad]"
        elif index in self.added_tokens_decoder:
            return f"{self.added_tokens_decpder[index]}"
        return self.sp_model.IdToPiece(index - self.num_image_tokens).replace("<n>", "\n")

    def convert_tokens_to_ids(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._convert_token_to_id(tokens)
        else:
            return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        if not isinstance(ids, (list, tuple)):
            return self._convert_id_to_token(ids)
        tokens = [self._convert_id_to_token(_id) for _id in ids]
        return tokens

    def convert_ids_to_string(self, ids):
        return self.sp_model.Decode(ids).replace("<n>", "\n")

    def save_resources(self, save_directory):
        for name, file_name in self.resource_files_names.items():
            save_path = os.path.join(save_directory, file_name)
            try:
                shutil.copyfile(getattr(self, "_%s" % name), save_path)
            except shutil.SameFileError:
                pass

    def contains_sentence_end(self, idx):
        return idx in self.sentence_end_decoder

    @property
    def eos_token_id(self):
        return self.added_tokens_encoder["[eos]"]

    @property
    def sop_token_id(self):
        return self.added_tokens_encoder["[sop]"]

    @property
    def eop_token_id(self):
        return self.added_tokens_encoder["[eop]"]

    @property
    def gmask_token_id(self):
        return self.convert_tokens_to_ids("[gMASK]")

    @property
    def smask_token_id(self):
        return self.convert_tokens_to_ids("[sMASK]")

    @property
    def mask_token_id(self):
        return self.convert_tokens_to_ids("[MASK]")
