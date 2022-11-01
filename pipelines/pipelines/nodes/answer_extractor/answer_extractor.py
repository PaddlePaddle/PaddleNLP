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
import os
import math
from multiprocessing import cpu_count
from tqdm import tqdm
import json

import paddle
from paddlenlp.taskflow.utils import download_file
from pipelines.nodes.answer_extractor import UIEComponent
from paddlenlp.utils.env import PPNLP_HOME


class AnswerExtractor(UIEComponent):
    """
    Universal Information Extraction Component. 
    """
    resource_files_urls = {
        "uie-base-answer-extractor-v1": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/pipelines/answer_generator/uie-base-answer-extractor/uie-base-answer-extractor-v1/model_state.pdparams",
                "c8619f631a0c20434199840d34bb8b8c"
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/pipelines/answer_generator/uie-base-answer-extractor/uie-base-answer-extractor-v1/model_config.json",
                "74f033ab874a1acddb3aec9b9c4d9cde"
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/pipelines/answer_generator/uie-base-answer-extractor/uie-base-answer-extractor-v1/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8"
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/pipelines/answer_generator/uie-base-answer-extractor/uie-base-answer-extractor-v1/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec"
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/pipelines/answer_generator/uie-base-answer-extractor/uie-base-answer-extractor-v1/tokenizer_config.json",
                "3e623b57084882fd73e17f544bdda47d"
            ]
        },
    }

    def __init__(self,
                 model='uie-base-answer-extractor',
                 schema=['答案'],
                 task_path=None,
                 device="gpu",
                 schema_lang="zh",
                 max_seq_len=512,
                 batch_size=64,
                 split_sentence=False,
                 position_prob=0.01,
                 lazy_load=False,
                 num_workers=0,
                 use_faster=False):
        paddle.set_device(device)
        if model in ['uie-m-base', 'uie-m-large']:
            self._multilingual = True
            self.resource_files_names[
                'sentencepiece_model_file'] = "sentencepiece.bpe.model"
        else:
            self._multilingual = False
            if 'sentencepiece_model_file' in self.resource_files_names.keys():
                del self.resource_files_names['sentencepiece_model_file']
        self._schema_tree = None
        self.set_schema(schema)

        self._is_en = True if model in ['uie-base-en'
                                        ] or schema_lang == 'en' else False
        #### task parameters
        self.model = model
        self.task = None
        self._priority_path = None
        self._usage = ""
        self._model = None
        self._input_spec = None
        self._config = None
        self._custom_model = False
        self._param_updated = False
        self._num_threads = math.ceil(cpu_count() / 2)
        self._infer_precision = 'fp32'
        self._predictor_type = 'paddle-inference'
        self._home_path = PPNLP_HOME
        self._task_flag = self.model
        if task_path:
            self._task_path = task_path
            self._custom_model = True
        else:
            self._task_path = os.path.join(PPNLP_HOME, "pipelines",
                                           "unsupervised_question_answering",
                                           self.model)

        self._check_task_files()
        self._check_predictor_type()
        self._get_inference_model()
        self._construct_tokenizer(model)

        self._schema = schema
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._split_sentence = split_sentence
        self._position_prob = position_prob
        self._lazy_load = lazy_load
        self._num_workers = num_workers
        self.use_faster = use_faster

    def answer_generation_from_paragraphs(self,
                                          paragraphs,
                                          model=None,
                                          max_answer_candidates=5,
                                          wf=None):
        """Generate answer from given paragraphs."""
        result = []
        buffer = []
        i = 0
        len_paragraphs = len(paragraphs)
        for paragraph_tobe in tqdm(paragraphs):
            buffer.append(paragraph_tobe)
            if len(buffer) == self._batch_size or (i + 1) == len_paragraphs:
                predicts = self.model_run(buffer)
                paragraph_list = buffer
                buffer = []
                for predict_dict, paragraph in zip(predicts, paragraph_list):
                    answers = []
                    probabilitys = []
                    for prompt in self._schema:
                        if prompt in predict_dict:
                            answer_dicts = predict_dict[prompt]
                            answers += [
                                answer_dict['text']
                                for answer_dict in answer_dicts
                            ]
                            probabilitys += [
                                answer_dict['probability']
                                for answer_dict in answer_dicts
                            ]
                        else:
                            answers += []
                            probabilitys += []
                    candidates = sorted(list(
                        set([(a, p) for a, p in zip(answers, probabilitys)])),
                                        key=lambda x: -x[1])
                    if len(candidates) > max_answer_candidates:
                        candidates = candidates[:max_answer_candidates]
                    outdict = {
                        'context': paragraph,
                        'answer_candidates': candidates,
                    }
                    if wf:
                        wf.write(json.dumps(outdict, ensure_ascii=False) + "\n")
                    result.append(outdict)
            i += 1
        return result

    def model_run(self, str_list):
        preprocessed = self._preprocess(str_list)
        generated = self._run_model(preprocessed)
        postprocessed = self._postprocess(generated)
        return postprocessed

    def run(self, meta):
        synthetic_context_answer_pairs = self.answer_generation_from_paragraphs(
            meta)
        results = {"ca_pairs": synthetic_context_answer_pairs}
        return results, "output_1"
