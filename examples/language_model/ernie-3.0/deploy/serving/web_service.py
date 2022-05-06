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
from paddle_serving_server.web_service import WebService, Op
from faster_tokenizers import ErnieFasterTokenizer, models

from numpy import array

import logging
import numpy as np

_LOGGER = logging.getLogger()

class ErnieOp(Op):
    def init_op(self):
        vocab = models.WordPiece.get_vocab_from_file("/vdb1/home/heliqi/.paddlenlp/models/ernie-3.0-tiny/ernie_3.0_tiny_vocab.txt")
        self.tokenizer = ErnieFasterTokenizer(vocab, max_sequence_len=128)

    def preprocess(self, input_dicts, data_id, log_id):
        print("input_dicts:", input_dicts)
        print("data_id:", data_id)
        print("log_id:", log_id)
        
        # convert input format
        (_, input_dict), = input_dicts.items()
        data = input_dict["sentence"]
        if isinstance(data, str) and "array(" in data:
            data = eval(data)
        else:
            _LOGGER.error("input value  {}is not supported.".format(data))

        data = [i.decode('utf-8') for i in data]
        data = self.tokenizer.encode_batch(data)
        input_ids = [i.get_ids() for i in data]
        token_type_ids = [i.get_type_ids() for i in data]

        return {"input_ids": np.array(input_ids, dtype="int64"), "token_type_ids": np.array(token_type_ids, dtype="int64")}, False, None, ""

class Ernie3Service(WebService):
    def get_pipeline_response(self, read_op):
        return ErnieOp(name="ernie", input_ops=[read_op])

ocr_service = Ernie3Service(name="erine3")
ocr_service.prepare_pipeline_config("config.yml")
ocr_service.run_service()
