# coding:utf-8
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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


class QuestionGenerator(BaseComponent):
    """
    Question Generator based on Unimo Text.
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
        "vocab_file": "vocab.txt",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json",
    }

    resource_files_urls = {
        "unimo-text-1.0-question-generator": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/pipelines/question_generator/unimo-text-1.0-question-generator-v1/model_state.pdparams",
                "856a2980f83dc227a8fed4ecd730696d",
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/pipelines/question_generator/unimo-text-1.0-question-generator-v1/model_config.json",
                "b5bab534683d9f0ef82fc84803ee6f3d",
            ],
            "vocab_file": [
                "https://bj.bcebos.com/paddlenlp/pipelines/question_generator/unimo-text-1.0-question-generator-v1/vocab.txt",
                "ea3f8a8cc03937a8df165d2b507c551e",
            ],
            "special_tokens_map": [
                "https://bj.bcebos.com/paddlenlp/pipelines/question_generator/unimo-text-1.0-question-generator-v1/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://bj.bcebos.com/paddlenlp/pipelines/question_generator/unimo-text-1.0-question-generator-v1/tokenizer_config.json",
                "ef261f5d413a46ed1d6f071aed6fb345",
            ],
        },
    }

    return_no_answers: bool
    outgoing_edges = 1
    query_count = 0
    query_time = 0

    def __init__(
        self,
        model="unimo-text-1.0-question-generation",
        task_path=None,
        device="gpu",
        batch_size=16,
        output_scores=True,
        is_select_from_num_return_sequences=False,
        max_length=50,
        decode_strategy="sampling",
        temperature=1.0,
        top_k=5,
        top_p=1.0,
        num_beams=6,
        num_beam_groups=1,
        diversity_rate=0.0,
        num_return_sequences=1,
        template=1,
    ):
        paddle.set_device(device)
        self.model = model
        self._from_taskflow = False
        self._custom_model = False
        if task_path:
            self._task_path = task_path
            self._custom_model = True
        else:
            if model in [
                "unimo-text-1.0",
                "unimo-text-1.0-dureader_qg",
                "unimo-text-1.0-question-generation",
                "unimo-text-1.0-question-generation-dureader_qg",
            ]:
                self._task_path = None
                self._from_taskflow = True
            else:
                self._task_path = os.path.join(PPNLP_HOME, "pipelines", "unsupervised_question_answering", self.model)
                self._check_task_files()
                self.model = "unimo-text-1.0"
        self.num_return_sequences = num_return_sequences
        self.batch_size = batch_size
        if self._from_taskflow:
            self.question_generation = Taskflow(
                "question_generation",
                model=self.model if self._from_taskflow else "unimo-text-1.0",
                output_scores=True,
                max_length=max_length,
                is_select_from_num_return_sequences=is_select_from_num_return_sequences,
                num_return_sequences=num_return_sequences,
                batch_size=batch_size,
                decode_strategy=decode_strategy,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_rate=diversity_rate,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                template=1,
                device_id=0 if device == "gpu" else -1,
            )
        else:
            self.question_generation = Taskflow(
                "question_generation",
                model=self.model if self._from_taskflow else "unimo-text-1.0",
                task_path=self._task_path,
                output_scores=True,
                max_length=max_length,
                is_select_from_num_return_sequences=is_select_from_num_return_sequences,
                num_return_sequences=num_return_sequences,
                batch_size=batch_size,
                decode_strategy=decode_strategy,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_rate=diversity_rate,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                template=1,
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

    def create_question(
        self, json_file_or_pair_list, out_json=None, num_return_sequences=1, all_sample_num=None, batch_size=8
    ):
        if out_json:
            wf = open(out_json, "w", encoding="utf-8")
        if isinstance(json_file_or_pair_list, list):
            all_lines = json_file_or_pair_list
        else:
            rf = open(json_file_or_pair_list, "r", encoding="utf-8")
            all_lines = []
            for json_line in rf:
                line_dict = json.loads(json_line)
                all_lines.append(line_dict)
            rf.close()
        num_all_lines = len(all_lines)
        output = []
        context_buffer = []
        answer_buffer = []
        answer_probability_buffer = []
        true_question_buffer = []
        i = 0
        for index, line_dict in enumerate(tqdm(all_lines)):
            if "question" in line_dict:
                q = line_dict["question"]
            else:
                q = ""
            c = line_dict["context"]
            assert "answer_candidates" in line_dict
            answers = line_dict["answer_candidates"]
            if not answers:
                continue
            for j, pair in enumerate(answers):
                a, p = pair
                context_buffer += [c]
                answer_buffer += [a]
                answer_probability_buffer += [p]
                true_question_buffer += [q]
                if (
                    (i + 1) % batch_size == 0
                    or (all_sample_num and (i + 1) == all_sample_num)
                    or ((index + 1) == num_all_lines and j == len(answers) - 1)
                ):
                    result_buffer = self.question_generation(
                        [
                            {"context": context, "answer": answer}
                            for context, answer in zip(context_buffer, answer_buffer)
                        ]
                    )
                    (
                        context_buffer_temp,
                        answer_buffer_temp,
                        answer_probability_buffer_temp,
                        true_question_buffer_temp,
                    ) = ([], [], [], [])
                    for context, answer, answer_probability, true_question in zip(
                        context_buffer, answer_buffer, answer_probability_buffer, true_question_buffer
                    ):
                        context_buffer_temp += [context] * num_return_sequences
                        answer_buffer_temp += [answer] * num_return_sequences
                        answer_probability_buffer_temp += [answer_probability] * num_return_sequences
                        true_question_buffer_temp += [true_question] * num_return_sequences
                    result_one_two_buffer = [(one, two) for one, two in zip(result_buffer[0], result_buffer[1])]
                    for context, answer, answer_probability, true_question, result in zip(
                        context_buffer_temp,
                        answer_buffer_temp,
                        answer_probability_buffer_temp,
                        true_question_buffer_temp,
                        result_one_two_buffer,
                    ):
                        fake_quesitons_tokens = [result[0]]
                        fake_quesitons_scores = [result[1]]
                        for fake_quesitons_token, fake_quesitons_score in zip(
                            fake_quesitons_tokens, fake_quesitons_scores
                        ):
                            out_dict = {
                                "context": context,
                                "synthetic_answer": answer,
                                "synthetic_answer_probability": answer_probability,
                                "synthetic_question": fake_quesitons_token,
                                "synthetic_question_probability": fake_quesitons_score,
                                "true_question": true_question,
                            }
                            if out_json:
                                wf.write(json.dumps(out_dict, ensure_ascii=False) + "\n")
                            output.append(out_dict)
                    context_buffer = []
                    answer_buffer = []
                    true_question_buffer = []
                if all_sample_num and (i + 1) >= all_sample_num:
                    break
                i += 1
        if out_json:
            wf.close()
        return output

    def run(self, ca_pairs):
        print("createing synthetic question-answer pairs...")
        synthetic_context_answer_question_triples = self.create_question(
            ca_pairs, None, self.num_return_sequences, None, self.batch_size
        )
        print("create synthetic question-answer pairs successfully!")
        results = {"cqa_triples": synthetic_context_answer_question_triples}
        return results, "output_1"
