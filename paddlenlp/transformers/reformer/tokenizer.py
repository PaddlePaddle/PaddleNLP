# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.utils import try_import
from ..albert.tokenizer import AlbertEnglishTokenizer


class ReformerTokenizer(AlbertEnglishTokenizer):
    resource_files_names = {"sentencepiece_model_file": "spiece.model", }
    pretrained_resource_files_map = {
        "sentencepiece_model_file": {
            "reformer-crime-and-punishment":
            "http://paddlenlp.bj.bcebos.com/models/transformers/reformer/reformer-crime-and-punishment/spiece.model",
        },
    }

    pretrained_init_configuration = {
        "reformer-crime-and-punishment": {
            "do_lower_case": False
        },
    }

    def __init__(self,
                 sentencepiece_model_file,
                 do_lower_case=False,
                 remove_space=True,
                 keep_accents=False,
                 eos_token="</s>",
                 unk_token="<unk>",
                 **kwargs):

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.sentencepiece_model_file = sentencepiece_model_file

        spm = try_import("sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sentencepiece_model_file)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return len(token_ids_0) * [0] + len(token_ids_1) * [1]
