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
from shutil import copyfile
from typing import Any, Dict, List

import sentencepiece as spm

from .. import PretrainedTokenizer

__all__ = ["GLM130BTokenizer"]


class GLM130BTokenizer(PretrainedTokenizer):
    resource_files_names: Dict[str, str] = {"model_file": "ice_text.model"}
    pretrained_resource_files_map: Dict[str, Dict[str, str]] = {"model _file": {"glm-130b": None}}
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {"glm-130b": {}}

    def __init__(self, model_file, max_len=512, ignore_linebreak=True, **kwargs):
        self._model_file = model_file
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
        self.num_tokens = 150000
        self.add_tokens(["MASK", "gMASK", "sMASK", "eod", "sop", "eop", "ENC", "dBLOCK"])
        self.sentence_end_decoder = {
            20007: ".",
            20031: "？",
            20035: "！",
            20027: "；",
            20012: ":",
            83823: "。",
            145670: "…",
        }
        self.added_tokens_encoder["eos"] = 20002
        self.added_tokens_decoder[20002] = "</s>"

    @property
    def vocab_size(self) -> int:
        return self.num_tokens

    def _tokenize(self, text: str) -> List[str]:
        if not self.ignore_linebreak:
            text = text.replace("\n", "<n>")
        self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token.replace("\n", "<n>"))

    def _convert_id_to_token(self, index):
        return self.sp_model.IdToPiece(index).replace("<n>", "\n")

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
            copyfile(getattr(self, "_%s" % name), save_path)

    def contains_sentence_end(self, idx):
        return idx in self.sentence_end_decoder

    @property
    def eod(self):
        return self.added_tokens_encoder["eos"]
