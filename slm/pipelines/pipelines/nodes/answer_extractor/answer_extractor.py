# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
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
import json
import os

import paddle
from paddle.dataset.common import md5file
from pipelines.nodes.base import BaseComponent
from tqdm import tqdm

from paddlenlp import Taskflow
from paddlenlp.taskflow.utils import download_file
from paddlenlp.utils.env import PPNLP_HOME


class AnswerExtractor(BaseComponent):
    """
    Answer Extractor based on Universal Information Extraction.
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
        "vocab_file": "vocab.txt",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json",
    }

    resource_files_urls = {
        "uie-base-answer-extractor": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/pipelines/answer_generator/uie-base-answer-extractor/uie-base-answer-extractor-v1/model_state.pdparams",
                "c8619f631a0c20434199840d34bb8b8c",
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/pipelines/answer_generator/uie-base-answer-extractor/uie-base-answer-extractor-v1/model_config.json",
                "74f033ab874a1acddb3aec9b9c4d9cde",
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/pipelines/answer_generator/uie-base-answer-extractor/uie-base-answer-extractor-v1/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8",
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/pipelines/answer_generator/uie-base-answer-extractor/uie-base-answer-extractor-v1/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/pipelines/answer_generator/uie-base-answer-extractor/uie-base-answer-extractor-v1/tokenizer_config.json",
                "3e623b57084882fd73e17f544bdda47d",
            ],
        },
    }

    return_no_answers: bool
    outgoing_edges = 1
    query_count = 0
    query_time = 0

    def __init__(
        self,
        model="uie-base-answer-extractor",
        schema=["答案"],
        task_path=None,
        device="gpu",
        batch_size=64,
        position_prob=0.01,
        max_answer_candidates=5,
    ):
        paddle.set_device(device)
        self.model = model
        self._from_taskflow = False
        self._custom_model = False
        if task_path:
            self._task_path = task_path
            self._custom_model = True
        else:
            if model in ["uie-base"]:
                self._task_path = None
                self._from_taskflow = True
            else:
                self._task_path = os.path.join(PPNLP_HOME, "pipelines", "unsupervised_question_answering", self.model)
                self._check_task_files()
        self.batch_size = batch_size
        self.max_answer_candidates = max_answer_candidates
        self.schema = schema
        self.answer_generator = Taskflow(
            "information_extraction",
            model=self.model if self._from_taskflow else "uie-base",
            schema=schema,
            task_path=self._task_path,
            batch_size=batch_size,
            position_prob=position_prob,
            device_id=0 if device == "gpu" else -1,
        )

    def _check_task_files(self):
        """
        Check files required by the task.
        """
        for file_id, file_name in self.resource_files_names.items():
            path = os.path.join(self._task_path, file_name)
            url = self.resource_files_urls[self.model][file_id][0]
            md5 = self.resource_files_urls[self.model][file_id][1]

            downloaded = True
            if not os.path.exists(path):
                downloaded = False
            else:
                if not self._custom_model:
                    if os.path.exists(path):
                        # Check whether the file is updated
                        if not md5file(path) == md5:
                            downloaded = False
                            if file_id == "model_state":
                                self._param_updated = True
                    else:
                        downloaded = False
            if not downloaded:
                download_file(self._task_path, file_name, url, md5)

    def answer_generation_from_paragraphs(
        self, paragraphs, batch_size=16, model=None, max_answer_candidates=5, schema=None, wf=None
    ):
        """Generate answer from given paragraphs."""
        result = []
        buffer = []
        i = 0
        len_paragraphs = len(paragraphs)
        for paragraph_tobe in tqdm(paragraphs):
            buffer.append(paragraph_tobe)
            if len(buffer) == batch_size or (i + 1) == len_paragraphs:
                predicts = model(buffer)
                paragraph_list = buffer
                buffer = []
                for predict_dict, paragraph in zip(predicts, paragraph_list):
                    answers = []
                    probabilitys = []
                    for prompt in schema:
                        if prompt in predict_dict:
                            answer_dicts = predict_dict[prompt]
                            answers += [answer_dict["text"] for answer_dict in answer_dicts]
                            probabilitys += [answer_dict["probability"] for answer_dict in answer_dicts]
                        else:
                            answers += []
                            probabilitys += []
                    candidates = sorted(
                        list(set([(a, p) for a, p in zip(answers, probabilitys)])), key=lambda x: -x[1]
                    )
                    if len(candidates) > max_answer_candidates:
                        candidates = candidates[:max_answer_candidates]
                    outdict = {
                        "context": paragraph,
                        "answer_candidates": candidates,
                    }
                    if wf:
                        wf.write(json.dumps(outdict, ensure_ascii=False) + "\n")
                    result.append(outdict)
            i += 1
        return result

    def run(self, meta):
        print("createing synthetic answers...")
        synthetic_context_answer_pairs = self.answer_generation_from_paragraphs(
            meta,
            batch_size=self.batch_size,
            model=self.answer_generator,
            max_answer_candidates=self.max_answer_candidates,
            schema=self.schema,
            wf=None,
        )
        results = {"ca_pairs": synthetic_context_answer_pairs}
        return results, "output_1"
