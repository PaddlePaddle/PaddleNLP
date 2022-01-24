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

import logging
import numpy as np
import sys

from paddle_serving_server.web_service import WebService, Op

_LOGGER = logging.getLogger()


class PPMiniLMOp(Op):
    def init_op(self):
        from paddlenlp.transformers import PPMiniLMTokenizer
        self.tokenizer = PPMiniLMTokenizer.from_pretrained('ppminilm-6l-768h')

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        feed_dict = {}
        feed_dict['text'] = list(input_dict.values())
        return feed_dict, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        new_dict = {}
        new_dict["logits"] = str(fetch_dict["logits"].tolist())
        return new_dict, None, ""


class PPMiniLMService(WebService):
    def get_pipeline_response(self, read_op):
        ppminilm_op = PPMiniLMOp(name="ppminilm", input_ops=[read_op])
        return ppminilm_op


ppminilm_service = PPMiniLMService(name="ppminilm")
ppminilm_service.prepare_pipeline_config("config_nlp.yml")
ppminilm_service.run_service()
